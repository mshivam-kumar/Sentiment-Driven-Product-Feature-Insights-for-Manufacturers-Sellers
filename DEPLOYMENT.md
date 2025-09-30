# 🚀 Deployment Guide

This guide covers multiple deployment options for the Sentiment Analysis App with RAG functionality.

## 📋 Prerequisites

### Required Tools
- **Docker** (for containerization)
- **Git** (for version control)
- **Python 3.11+** (for local development)

### Optional Tools
- **AWS CLI** (for AWS deployment)
- **Heroku CLI** (for Heroku deployment)
- **Terraform** (for infrastructure management)

## 🐳 Docker Deployment

### Local Docker Deployment

1. **Build the Docker image:**
   ```bash
   docker build -t sentiment-analysis-app:latest .
   ```

2. **Run the container:**
   ```bash
   docker run -d \
     --name sentiment-analysis-app \
     -p 8501:8501 \
     -e AWS_ACCESS_KEY_ID="your-access-key" \
     -e AWS_SECRET_ACCESS_KEY="your-secret-key" \
     -e AWS_DEFAULT_REGION="us-east-1" \
     sentiment-analysis-app:latest
   ```

3. **Access the app:**
   - Open http://localhost:8501 in your browser

### Docker Compose (Alternative)

Create a `docker-compose.yml` file:

```yaml
version: '3.8'
services:
  sentiment-analysis:
    build: .
    ports:
      - "8501:8501"
    environment:
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
      - AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION}
    restart: unless-stopped
```

Run with: `docker-compose up -d`

## ☁️ Cloud Deployment Options

### 1. AWS ECS Deployment

#### Prerequisites
- AWS CLI configured with appropriate permissions
- ECR repository created
- ECS cluster and service configured

#### Steps

1. **Build and push to ECR:**
   ```bash
   # Get AWS account ID
   ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
   
   # Login to ECR
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
   
   # Build and tag
   docker build -t sentiment-analysis-app .
   docker tag sentiment-analysis-app:latest $ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/sentiment-analysis-app:latest
   
   # Push to ECR
   docker push $ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/sentiment-analysis-app:latest
   ```

2. **Update ECS task definition:**
   - Replace `ACCOUNT_ID` in `infra/ecs-task-definition.json`
   - Update the image URI with your ECR repository

3. **Deploy to ECS:**
   ```bash
   aws ecs update-service --cluster sentiment-analysis-cluster --service sentiment-analysis-service --task-definition sentiment-analysis-task
   ```

### 2. Heroku Deployment

#### Prerequisites
- Heroku CLI installed
- Heroku account

#### Steps

1. **Login to Heroku:**
   ```bash
   heroku login
   ```

2. **Create Heroku app:**
   ```bash
   heroku create your-app-name
   ```

3. **Set environment variables:**
   ```bash
   heroku config:set AWS_ACCESS_KEY_ID="your-access-key"
   heroku config:set AWS_SECRET_ACCESS_KEY="your-secret-key"
   heroku config:set AWS_DEFAULT_REGION="us-east-1"
   ```

4. **Deploy:**
   ```bash
   git push heroku main
   ```

### 3. Streamlit Cloud Deployment

1. **Go to Streamlit Cloud:**
   - Visit https://share.streamlit.io/
   - Sign in with GitHub

2. **Deploy your app:**
   - Click "New app"
   - Select your repository
   - Set main file path: `dashboard/streamlit_app.py`
   - Add environment variables if needed
   - Click "Deploy"

## 🤖 Automated Deployment with GitHub Actions

### Setup GitHub Secrets

Add these secrets to your GitHub repository:

#### For AWS Deployment:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

#### For Heroku Deployment:
- `HEROKU_API_KEY`
- `HEROKU_APP_NAME`
- `HEROKU_EMAIL`

#### For Streamlit Cloud:
- `STREAMLIT_CLOUD_TOKEN`
- `STREAMLIT_APP_URL`

### GitHub Actions Workflow

The `.github/workflows/deploy.yml` file includes:

1. **Testing:** Runs pytest tests
2. **Building:** Creates Docker image
3. **AWS ECS:** Deploys to AWS ECS
4. **Streamlit Cloud:** Deploys to Streamlit Cloud
5. **Heroku:** Deploys to Heroku

### Triggering Deployment

- **Automatic:** Push to `main` or `master` branch
- **Manual:** Go to Actions tab → Deploy Sentiment Analysis App → Run workflow

## 🛠️ Using the Deployment Script

The `deploy.sh` script automates the deployment process:

```bash
# Make executable
chmod +x deploy.sh

# Run deployment script
./deploy.sh
```

The script will:
1. Check prerequisites
2. Build Docker image
3. Offer deployment options:
   - Local Docker
   - AWS ECS
   - Heroku
   - Streamlit Cloud
   - All of the above

## 🔧 Environment Variables

### Required for AWS Integration
```bash
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_DEFAULT_REGION=us-east-1
```

### Optional for Enhanced Features
```bash
# For RAG functionality
SENTENCE_TRANSFORMERS_CACHE=/tmp/transformers_cache

# For monitoring
LOG_LEVEL=INFO
```

## 📊 Monitoring and Health Checks

### Health Check Endpoints
- **Streamlit Health:** `http://your-app-url/_stcore/health`
- **Custom Health:** `http://your-app-url/health`

### Logs
- **Docker:** `docker logs sentiment-analysis-app`
- **AWS ECS:** Check CloudWatch logs
- **Heroku:** `heroku logs --tail`
- **Streamlit Cloud:** Check the logs tab in the dashboard

## 🚨 Troubleshooting

### Common Issues

1. **Port Conflicts:**
   - Ensure port 8501 is available
   - Use `-p 8502:8501` for different local port

2. **AWS Credentials:**
   - Verify AWS credentials are set correctly
   - Check IAM permissions for ECR and ECS

3. **Memory Issues:**
   - Increase Docker memory limit
   - Use larger ECS task definition

4. **RAG Dependencies:**
   - Ensure `sentence-transformers` is installed
   - Check if model downloads are working

### Performance Optimization

1. **Docker Image Size:**
   - Use multi-stage builds
   - Remove unnecessary dependencies

2. **Startup Time:**
   - Pre-download models in Docker image
   - Use smaller base images

3. **Memory Usage:**
   - Monitor container memory usage
   - Adjust ECS task memory allocation

## 🔄 CI/CD Pipeline

The GitHub Actions workflow provides:

1. **Automated Testing:** Runs on every push
2. **Multi-Platform Deployment:** AWS, Heroku, Streamlit Cloud
3. **Rollback Capability:** Easy rollback to previous versions
4. **Monitoring:** Health checks and logging

## 📈 Scaling

### Horizontal Scaling
- **AWS ECS:** Increase service desired count
- **Heroku:** Use Heroku Dynos
- **Streamlit Cloud:** Built-in scaling

### Vertical Scaling
- **AWS ECS:** Increase task CPU/memory
- **Heroku:** Upgrade dyno type
- **Docker:** Increase container resources

## 🛡️ Security Considerations

1. **Environment Variables:** Never commit secrets to Git
2. **AWS IAM:** Use least privilege principle
3. **Network Security:** Configure security groups properly
4. **HTTPS:** Use SSL certificates for production

## 📞 Support

For deployment issues:
1. Check the logs for error messages
2. Verify environment variables are set
3. Ensure all prerequisites are installed
4. Review the troubleshooting section above

---

**Happy Deploying! 🚀**