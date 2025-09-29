# 🚀 Deployment Guide

This guide covers multiple deployment options for the Sentiment-Driven Product Feature Insights application.

## 📋 Prerequisites

- Python 3.11+
- Git
- AWS CLI (for AWS deployment)
- Heroku CLI (for Heroku deployment)
- Streamlit Cloud account (for Streamlit Cloud deployment)

## 🎯 Deployment Options

### Option 1: Streamlit Cloud (Recommended for Quick Start)

Streamlit Cloud is the easiest way to deploy your Streamlit app.

#### Steps:
1. **Prepare your repository:**
   ```bash
   # Run the deployment script
   python deploy_models.py
   ```

2. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Add deployment package"
   git push origin main
   ```

3. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Select the `deployment/app.py` file
   - Set environment variables if needed
   - Click "Deploy"

#### Environment Variables:
- `API_BASE_URL`: Your AWS API Gateway URL (optional, defaults to the current one)

### Option 2: Heroku

Heroku provides a robust platform for web applications.

#### Steps:
1. **Install Heroku CLI:**
   ```bash
   # On Ubuntu/Debian
   curl https://cli-assets.heroku.com/install.sh | sh
   ```

2. **Login to Heroku:**
   ```bash
   heroku login
   ```

3. **Create Heroku app:**
   ```bash
   cd deployment
   heroku create your-app-name
   ```

4. **Set environment variables:**
   ```bash
   heroku config:set API_BASE_URL=https://f3157r5ca4.execute-api.us-east-1.amazonaws.com/dev
   ```

5. **Deploy:**
   ```bash
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   ```

### Option 3: AWS EC2/ECS

For production deployments with full control.

#### Steps:
1. **Launch EC2 instance:**
   ```bash
   # Use Amazon Linux 2 or Ubuntu
   aws ec2 run-instances \
     --image-id ami-0c02fb55956c7d316 \
     --instance-type t3.medium \
     --key-name your-key-pair \
     --security-group-ids sg-xxxxxxxxx
   ```

2. **Install dependencies:**
   ```bash
   sudo yum update -y
   sudo yum install -y python3 python3-pip git
   pip3 install -r requirements_deploy.txt
   ```

3. **Deploy application:**
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo/deployment
   streamlit run app.py --server.port=8501 --server.address=0.0.0.0
   ```

4. **Set up reverse proxy (Nginx):**
   ```bash
   sudo yum install -y nginx
   # Configure Nginx to proxy to Streamlit
   ```

### Option 4: Docker

Containerized deployment for any platform.

#### Steps:
1. **Create Dockerfile:**
   ```dockerfile
   FROM python:3.11-slim
   
   WORKDIR /app
   
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   
   COPY . .
   
   EXPOSE 8501
   
   CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **Build and run:**
   ```bash
   docker build -t sentiment-insights .
   docker run -p 8501:8501 sentiment-insights
   ```

## 🔧 GitHub Actions Setup

The repository includes GitHub Actions for automated deployment.

### Required Secrets:
Add these secrets to your GitHub repository:

1. **For Streamlit Cloud:**
   - `STREAMLIT_TOKEN`: Your Streamlit Cloud token
   - `STREAMLIT_APP_URL`: Your app URL

2. **For Heroku:**
   - `HEROKU_API_KEY`: Your Heroku API key
   - `HEROKU_EMAIL`: Your Heroku email

3. **For AWS:**
   - `AWS_ACCESS_KEY_ID`: Your AWS access key
   - `AWS_SECRET_ACCESS_KEY`: Your AWS secret key
   - `API_GATEWAY_ID`: Your API Gateway ID

4. **For Notifications:**
   - `SLACK_WEBHOOK`: Slack webhook URL (optional)

### Setting up Secrets:
1. Go to your GitHub repository
2. Click on "Settings" → "Secrets and variables" → "Actions"
3. Click "New repository secret"
4. Add each secret with the appropriate value

## 🌐 Making Your App Public

### 1. Streamlit Cloud (Easiest)
- Your app will be available at: `https://your-app-name.streamlit.app`
- Share this URL with anyone

### 2. Heroku
- Your app will be available at: `https://your-app-name.herokuapp.com`
- Share this URL with anyone

### 3. Custom Domain
- Purchase a domain from any registrar
- Configure DNS to point to your deployment
- Set up SSL certificate

## 📊 Monitoring and Maintenance

### Health Checks:
```bash
# Check if your app is running
curl https://your-app-url.com/health

# Check API endpoints
curl https://your-api-url.com/sentiment/product/B08JTNQFZY
```

### Logs:
```bash
# Heroku logs
heroku logs --tail

# Streamlit Cloud logs
# Available in the Streamlit Cloud dashboard
```

### Updates:
- Push changes to your main branch
- GitHub Actions will automatically deploy updates
- Monitor deployment status in the Actions tab

## 🔒 Security Considerations

1. **Environment Variables:**
   - Never commit secrets to your repository
   - Use environment variables for sensitive data

2. **API Keys:**
   - Rotate API keys regularly
   - Use IAM roles when possible

3. **Rate Limiting:**
   - Implement rate limiting for public APIs
   - Monitor usage and costs

## 🚨 Troubleshooting

### Common Issues:

1. **App won't start:**
   - Check Python version compatibility
   - Verify all dependencies are installed
   - Check logs for error messages

2. **Models not loading:**
   - Ensure spaCy model is downloaded
   - Check file paths and permissions
   - Verify model files are included in deployment

3. **API errors:**
   - Verify API Gateway URL is correct
   - Check AWS credentials and permissions
   - Monitor CloudWatch logs

4. **Performance issues:**
   - Consider upgrading instance size
   - Implement caching
   - Optimize model loading

### Getting Help:
- Check the logs in your deployment platform
- Review the GitHub Actions workflow runs
- Monitor AWS CloudWatch for Lambda errors
- Check the Streamlit Cloud dashboard for app status

## 📈 Scaling Considerations

### For High Traffic:
1. **Use AWS ECS/Fargate** for containerized deployment
2. **Implement caching** with Redis or ElastiCache
3. **Use CDN** for static assets
4. **Scale Lambda functions** with provisioned concurrency
5. **Implement database connection pooling**

### Cost Optimization:
1. **Use spot instances** for non-critical workloads
2. **Implement auto-scaling** based on demand
3. **Monitor and optimize** Lambda cold starts
4. **Use S3 lifecycle policies** for data archival

## 🎉 Success!

Once deployed, your app will be accessible to anyone with the URL. Share it with manufacturers, sellers, and other stakeholders to get feedback and improve the system.

Remember to:
- Monitor usage and performance
- Collect user feedback
- Regularly update dependencies
- Keep your models and data fresh
