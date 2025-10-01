terraform {
  required_version = ">= 1.3.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# NOTE: This infra directory is for learning/reference only.
# It is not wired to your current deployed stack and can be used
# as a scaffold if you want to recreate pieces from scratch.


