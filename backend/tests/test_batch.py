"""Tests for batch processing service."""

import pytest
from app.services.batch import CSVParser, CSVRow, CSVValidationError


@pytest.fixture
def parser():
    """Create CSV parser instance."""
    return CSVParser()


class TestCSVParserBasic:
    """Test basic CSV parsing."""
    
    def test_valid_csv_minimal(self, parser):
        """Test parsing CSV with only required columns."""
        csv_content = """filename,brand_name
label1.png,OLD TOM DISTILLERY
label2.png,JACK DANIELS"""
        
        rows, errors = parser.parse(csv_content)
        
        assert len(rows) == 2
        assert len(errors) == 0
        assert rows[0].filename == "label1.png"
        assert rows[0].brand_name == "OLD TOM DISTILLERY"
        assert rows[1].filename == "label2.png"
        assert rows[1].brand_name == "JACK DANIELS"
    
    def test_valid_csv_all_columns(self, parser):
        """Test parsing CSV with all columns."""
        csv_content = """filename,brand_name,class_type,abv_percent,net_contents_ml,has_warning
label1.png,OLD TOM DISTILLERY,Kentucky Bourbon,45,750,true
label2.png,JACK DANIELS,Tennessee Whiskey,40,1000,false"""
        
        rows, errors = parser.parse(csv_content)
        
        assert len(rows) == 2
        assert len(errors) == 0
        
        assert rows[0].class_type == "Kentucky Bourbon"
        assert rows[0].abv_percent == 45.0
        assert rows[0].net_contents_ml == 750.0
        assert rows[0].has_warning is True
        
        assert rows[1].has_warning is False
    
    def test_case_insensitive_columns(self, parser):
        """Test that column names are case-insensitive."""
        csv_content = """FILENAME,Brand_Name,CLASS_TYPE
label1.png,TEST,Bourbon"""
        
        rows, errors = parser.parse(csv_content)
        
        assert len(rows) == 1
        assert rows[0].filename == "label1.png"
        assert rows[0].brand_name == "TEST"
        assert rows[0].class_type == "Bourbon"


class TestCSVParserValidation:
    """Test CSV validation."""
    
    def test_missing_required_column(self, parser):
        """Test error when required column is missing."""
        csv_content = """filename,class_type
label1.png,Bourbon"""
        
        rows, errors = parser.parse(csv_content)
        
        assert len(rows) == 0
        assert len(errors) == 1
        assert "brand_name" in errors[0].message.lower()
    
    def test_empty_filename(self, parser):
        """Test error for empty filename."""
        csv_content = """filename,brand_name
,OLD TOM DISTILLERY"""
        
        rows, errors = parser.parse(csv_content)
        
        assert len(rows) == 0
        assert len(errors) == 1
        assert "filename" in errors[0].field.lower()
    
    def test_empty_brand_name(self, parser):
        """Test error for empty brand name."""
        csv_content = """filename,brand_name
label1.png,"""
        
        rows, errors = parser.parse(csv_content)
        
        assert len(rows) == 0
        assert len(errors) == 1
        assert "brand_name" in errors[0].field.lower()
    
    def test_invalid_abv_percent(self, parser):
        """Test error for invalid ABV."""
        csv_content = """filename,brand_name,abv_percent
label1.png,TEST,not_a_number"""
        
        rows, errors = parser.parse(csv_content)
        
        # Row is still included but with error
        assert len(rows) == 1
        assert len(errors) == 1
        assert rows[0].abv_percent is None
    
    def test_abv_out_of_range(self, parser):
        """Test error for ABV out of range."""
        csv_content = """filename,brand_name,abv_percent
label1.png,TEST,150"""
        
        rows, errors = parser.parse(csv_content)
        
        assert len(rows) == 1
        assert len(errors) == 1
        assert "between 0 and 100" in errors[0].message
        assert rows[0].abv_percent is None
    
    def test_negative_net_contents(self, parser):
        """Test error for negative net contents."""
        csv_content = """filename,brand_name,net_contents_ml
label1.png,TEST,-500"""
        
        rows, errors = parser.parse(csv_content)
        
        assert len(rows) == 1
        assert len(errors) == 1
        assert "positive" in errors[0].message.lower()
    
    def test_invalid_has_warning(self, parser):
        """Test error for invalid has_warning value."""
        csv_content = """filename,brand_name,has_warning
label1.png,TEST,maybe"""
        
        rows, errors = parser.parse(csv_content)
        
        assert len(rows) == 1
        assert len(errors) == 1
        assert "has_warning" in errors[0].field.lower()


class TestCSVParserHasWarning:
    """Test has_warning parsing."""
    
    def test_true_variants(self, parser):
        """Test true value variants."""
        for value in ["true", "True", "TRUE", "yes", "YES", "1"]:
            csv_content = f"""filename,brand_name,has_warning
label1.png,TEST,{value}"""
            
            rows, errors = parser.parse(csv_content)
            assert rows[0].has_warning is True, f"Failed for value: {value}"
    
    def test_false_variants(self, parser):
        """Test false value variants."""
        for value in ["false", "False", "FALSE", "no", "NO", "0"]:
            csv_content = f"""filename,brand_name,has_warning
label1.png,TEST,{value}"""
            
            rows, errors = parser.parse(csv_content)
            assert rows[0].has_warning is False, f"Failed for value: {value}"
    
    def test_default_true(self, parser):
        """Test has_warning defaults to true."""
        csv_content = """filename,brand_name
label1.png,TEST"""
        
        rows, errors = parser.parse(csv_content)
        assert rows[0].has_warning is True


class TestCSVParserEdgeCases:
    """Test edge cases."""
    
    def test_empty_csv(self, parser):
        """Test empty CSV file."""
        csv_content = ""
        
        rows, errors = parser.parse(csv_content)
        
        assert len(rows) == 0
        assert len(errors) == 1
    
    def test_header_only(self, parser):
        """Test CSV with header only."""
        csv_content = "filename,brand_name"
        
        rows, errors = parser.parse(csv_content)
        
        assert len(rows) == 0
        assert len(errors) == 0
    
    def test_extra_whitespace(self, parser):
        """Test that whitespace is trimmed."""
        csv_content = """filename,brand_name
  label1.png  ,  OLD TOM DISTILLERY  """
        
        rows, errors = parser.parse(csv_content)
        
        assert rows[0].filename == "label1.png"
        assert rows[0].brand_name == "OLD TOM DISTILLERY"
    
    def test_unknown_columns_ignored(self, parser):
        """Test that unknown columns are ignored."""
        csv_content = """filename,brand_name,unknown_column
label1.png,TEST,ignored_value"""
        
        rows, errors = parser.parse(csv_content)
        
        assert len(rows) == 1
        assert len(errors) == 0
    
    def test_mixed_valid_invalid_rows(self, parser):
        """Test mixed valid and invalid rows."""
        csv_content = """filename,brand_name,abv_percent
label1.png,TEST1,45
label2.png,,40
label3.png,TEST3,invalid
label4.png,TEST4,50"""
        
        rows, errors = parser.parse(csv_content)
        
        # 3 valid rows (label1, label3 with null ABV, label4)
        assert len(rows) == 3
        # 2 errors (empty brand_name, invalid abv)
        assert len(errors) == 2


class TestFilenameMatching:
    """Test filename matching between CSV and uploads."""
    
    def test_all_match(self, parser):
        """Test when all files match."""
        csv_content = """filename,brand_name
label1.png,TEST1
label2.png,TEST2"""
        
        rows, _ = parser.parse(csv_content)
        uploaded = ["label1.png", "label2.png"]
        
        matched, errors = parser.validate_filenames_match(rows, uploaded)
        
        assert len(matched) == 2
        assert len(errors) == 0
    
    def test_missing_upload(self, parser):
        """Test when uploaded file is missing."""
        csv_content = """filename,brand_name
label1.png,TEST1
label2.png,TEST2"""
        
        rows, _ = parser.parse(csv_content)
        uploaded = ["label1.png"]  # label2.png missing
        
        matched, errors = parser.validate_filenames_match(rows, uploaded)
        
        assert len(matched) == 1
        assert len(errors) == 1
        assert "label2.png" in errors[0].message
    
    def test_extra_upload(self, parser):
        """Test when extra file is uploaded without CSV entry."""
        csv_content = """filename,brand_name
label1.png,TEST1"""
        
        rows, _ = parser.parse(csv_content)
        uploaded = ["label1.png", "extra.png"]
        
        matched, errors = parser.validate_filenames_match(rows, uploaded)
        
        assert len(matched) == 1
        assert len(errors) == 1
        assert "extra.png" in errors[0].message


class TestCSVRowConversion:
    """Test CSVRow conversion."""
    
    def test_to_dict(self):
        """Test converting CSVRow to dict."""
        row = CSVRow(
            filename="test.png",
            brand_name="TEST",
            class_type="Bourbon",
            abv_percent=45.0,
            net_contents_ml=750.0,
            has_warning=True,
            row_number=2
        )
        
        d = row.to_dict()
        
        assert d["filename"] == "test.png"
        assert d["brand_name"] == "TEST"
        assert d["class_type"] == "Bourbon"
        assert d["abv_percent"] == 45.0
        assert d["net_contents_ml"] == 750.0
        assert d["has_warning"] is True
        # row_number should not be in dict
        assert "row_number" not in d


class TestLargeCSV:
    """Test handling of larger CSV files."""
    
    def test_many_rows(self, parser):
        """Test parsing many rows."""
        header = "filename,brand_name,abv_percent,net_contents_ml"
        rows = [f"label{i}.png,TEST{i},{40 + i % 20},{750}" for i in range(100)]
        csv_content = "\n".join([header] + rows)
        
        parsed_rows, errors = parser.parse(csv_content)
        
        assert len(parsed_rows) == 100
        assert len(errors) == 0


class TestDecimalValues:
    """Test decimal value parsing."""
    
    def test_decimal_abv(self, parser):
        """Test decimal ABV values."""
        csv_content = """filename,brand_name,abv_percent
label1.png,TEST,45.5
label2.png,TEST,12.75"""
        
        rows, errors = parser.parse(csv_content)
        
        assert rows[0].abv_percent == 45.5
        assert rows[1].abv_percent == 12.75
    
    def test_decimal_net_contents(self, parser):
        """Test decimal net contents values."""
        csv_content = """filename,brand_name,net_contents_ml
label1.png,TEST,750.0
label2.png,TEST,1000.5"""
        
        rows, errors = parser.parse(csv_content)
        
        assert rows[0].net_contents_ml == 750.0
        assert rows[1].net_contents_ml == 1000.5
