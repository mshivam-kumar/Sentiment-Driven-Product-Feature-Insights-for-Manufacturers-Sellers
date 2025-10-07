# 🚀 AWS Deployment Guide

## Overview
This guide will help you deploy the SellerIQ full-stack application to AWS using:
- **AWS ECS Fargate** - Serverless container orchestration
- **AWS ECR** - Container registry
- **AWS ALB** - Application Load Balancer
- **GitHub Actions** - CI/CD pipeline
- **Terraform** - Infrastructure as Code

## 📋 Prerequisites

### 1. AWS Account Setup
- AWS Account with appropriate permissions
- AWS CLI installed and configured
- Docker installed locally

### 2. GitHub Repository Setup
- Repository pushed to GitHub
- GitHub Actions enabled

## 🔧 Step 1: Configure AWS CLI

### Install AWS CLI (if not already installed):
```bash
# Ubuntu/Debian
sudo apt-get install awscli

# Or using pip
pip install awscli
```

### Configure AWS credentials:
```bash
aws configure
```
Enter your:
- AWS Access Key ID
- AWS Secret Access Key
- Default region (e.g., `us-east-1`)
- Default output format (e.g., `json`)

### Verify configuration:
```bash
aws sts get-caller-identity
```

## 🔑 Step 2: Configure GitHub Secrets

Go to your GitHub repository → Settings → Secrets and variables → Actions

Add these secrets:

### Required Secrets:
```
AWS_ACCESS_KEY_ID=your-access-key-id
AWS_SECRET_ACCESS_KEY=your-secret-access-key
AWS_REGION=us-east-1
ECR_REPOSITORY_BACKEND=selleriq-backend
ECR_REPOSITORY_FRONTEND=selleriq-frontend
ECS_CLUSTER_NAME=selleriq-cluster
ECS_SERVICE_BACKEND_NAME=selleriq-backend-service
ECS_SERVICE_FRONTEND_NAME=selleriq-frontend-service
```

## 🏗️ Step 3: Deploy Infrastructure with Terraform

### Navigate to terraform directory:
```bash
cd terraform
```

### Initialize Terraform:
```bash
terraform init
```

### Review the infrastructure plan:
```bash
terraform plan
```

### Deploy the infrastructure:
```bash
terraform apply
```

This will create:
- VPC with public/private subnets
- ECS Cluster
- ECR Repositories
- Application Load Balancer
- Security Groups
- IAM Roles

## 🚀 Step 4: Deploy Application

### Option A: Automatic Deployment (Recommended)
Simply push to the main branch:
```bash
git add .
git commit -m "Deploy to AWS"
git push origin main
```

GitHub Actions will automatically:
1. Build Docker images
2. Push to ECR
3. Deploy to ECS
4. Update ALB

### Option B: Manual Deployment
```bash
# Build and push backend image
docker build -t selleriq-backend .
docker tag selleriq-backend:latest <account-id>.dkr.ecr.<region>.amazonaws.com/selleriq-backend:latest
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/selleriq-backend:latest

# Build and push frontend image
docker build -t selleriq-frontend ./frontend
docker tag selleriq-frontend:latest <account-id>.dkr.ecr.<region>.amazonaws.com/selleriq-frontend:latest
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/selleriq-frontend:latest
```

## 🌐 Step 5: Access Your Application

After deployment, you'll get:
- **Frontend URL**: `http://your-alb-dns-name/`
- **Backend API**: `http://your-alb-dns-name/api/`
- **API Documentation**: `http://your-alb-dns-name/docs`

## 📊 Monitoring and Management

### View ECS Services:
```bash
aws ecs list-services --cluster selleriq-cluster
```

### View Application Logs:
```bash
aws logs describe-log-groups --log-group-name-prefix /ecs/selleriq
```

### Scale Services:
```bash
aws ecs update-service --cluster selleriq-cluster --service selleriq-backend-service --desired-count 2
```

## 💰 Cost Estimation

### Monthly Costs (Approximate):
- **ECS Fargate**: $15-30 (depending on usage)
- **ALB**: $16
- **ECR**: $1-5 (storage)
- **Data Transfer**: $5-15
- **Total**: ~$40-70/month

### Cost Optimization:
- Use Spot instances for non-production
- Set up auto-scaling
- Monitor with CloudWatch

## 🔧 Troubleshooting

### Common Issues:

1. **Container fails to start**:
   - Check ECS task logs
   - Verify environment variables
   - Check security group rules

2. **ALB health checks failing**:
   - Verify health check endpoints
   - Check security group rules
   - Ensure containers are listening on correct ports

3. **GitHub Actions failing**:
   - Verify AWS credentials
   - Check ECR repository permissions
   - Review GitHub Actions logs

### Useful Commands:
```bash
# Check ECS service status
aws ecs describe-services --cluster selleriq-cluster --services selleriq-backend-service

# View task logs
aws logs get-log-events --log-group-name /ecs/selleriq-backend --log-stream-name <stream-name>

# Update service
aws ecs update-service --cluster selleriq-cluster --service selleriq-backend-service --force-new-deployment
```

## 🎯 Next Steps

1. **Set up monitoring** with CloudWatch
2. **Configure auto-scaling** based on CPU/memory
3. **Set up SSL/TLS** with AWS Certificate Manager
4. **Configure custom domain** with Route 53
5. **Set up backup strategies** for data persistence

## 📞 Support

If you encounter issues:
1. Check the GitHub Actions logs
2. Review ECS service events
3. Check CloudWatch logs
4. Verify security group configurations

---

**🎉 Congratulations!** Your SellerIQ application is now running on AWS with full CI/CD pipeline!
