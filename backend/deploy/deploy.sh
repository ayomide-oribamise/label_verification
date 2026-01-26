#!/bin/bash
# Deploy Label Verification API to Azure Container Apps
# Usage: ./deploy.sh [resource-group] [location]

set -e

# Configuration
RESOURCE_GROUP="${1:-label-verification-rg}"
LOCATION="${2:-eastus}"
APP_NAME="label-verification-api"
ACR_NAME="labelverificationacr"
IMAGE_NAME="label-verification-backend"
IMAGE_TAG="${3:-latest}"

echo "========================================"
echo "Label Verification API - Azure Deployment"
echo "========================================"
echo "Resource Group: $RESOURCE_GROUP"
echo "Location: $LOCATION"
echo "App Name: $APP_NAME"
echo "========================================"

# Check if logged in to Azure
echo "Checking Azure login..."
if ! az account show &> /dev/null; then
    echo "Not logged in to Azure. Running 'az login'..."
    az login
fi

# Create resource group if it doesn't exist
echo "Creating resource group..."
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" --output none

# Create Azure Container Registry if it doesn't exist
echo "Creating Azure Container Registry..."
az acr create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$ACR_NAME" \
    --sku Basic \
    --admin-enabled true \
    --output none 2>/dev/null || echo "ACR already exists"

# Get ACR credentials
echo "Getting ACR credentials..."
ACR_LOGIN_SERVER=$(az acr show --name "$ACR_NAME" --query loginServer -o tsv)
ACR_USERNAME=$(az acr credential show --name "$ACR_NAME" --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name "$ACR_NAME" --query passwords[0].value -o tsv)

# Build and push container image
echo "Building container image..."
cd "$(dirname "$0")/.."

echo "Logging in to ACR..."
az acr login --name "$ACR_NAME"

echo "Building and pushing image..."
docker build -t "$ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG" .
docker push "$ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG"

# Deploy using Bicep template
echo "Deploying to Azure Container Apps..."
DEPLOYMENT_OUTPUT=$(az deployment group create \
    --resource-group "$RESOURCE_GROUP" \
    --template-file "deploy/azure-container-app.bicep" \
    --parameters \
        appName="$APP_NAME" \
        containerImage="$ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG" \
        containerRegistry="$ACR_LOGIN_SERVER" \
        registryUsername="$ACR_USERNAME" \
        registryPassword="$ACR_PASSWORD" \
        minReplicas=1 \
    --query properties.outputs)

# Extract the app URL
APP_URL=$(echo "$DEPLOYMENT_OUTPUT" | jq -r '.appUrl.value')

echo ""
echo "========================================"
echo "Deployment Complete!"
echo "========================================"
echo "App URL: $APP_URL"
echo ""
echo "Test endpoints:"
echo "  Health: $APP_URL/api/v1/health"
echo "  Docs:   $APP_URL/docs"
echo "========================================"

# Wait for app to be ready and test health
echo ""
echo "Waiting for app to be ready (this may take 1-2 minutes)..."
sleep 30

for i in {1..10}; do
    if curl -s "$APP_URL/api/v1/health" | grep -q "healthy"; then
        echo "✅ App is healthy and ready!"
        curl -s "$APP_URL/api/v1/health" | jq .
        exit 0
    fi
    echo "  Attempt $i/10 - waiting..."
    sleep 15
done

echo "⚠️  App may still be starting. Check manually: $APP_URL/api/v1/health"
