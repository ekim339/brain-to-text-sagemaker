"""
Debug script to check SageMaker resource availability
"""
import boto3
import sagemaker

# Get session info
sess = sagemaker.Session()
region = sess.boto_region_name

print("="*80)
print("SageMaker Environment Check")
print("="*80)
print(f"Region: {region}")
print(f"Default bucket: {sess.default_bucket()}")

# Check execution role
try:
    role = sagemaker.get_execution_role()
    print(f"Execution role: {role}")
except Exception as e:
    print(f"ERROR getting execution role: {e}")
    print("\nYou may need to create a SageMaker execution role.")
    print("Go to: https://console.aws.amazon.com/iam/home#/roles")
    print("Create a role with 'SageMaker' use case and AmazonSageMakerFullAccess policy")

# Check available instance types
print("\n" + "="*80)
print("Checking common GPU instance availability...")
print("="*80)

client = boto3.client('sagemaker', region_name=region)

# Test a few common instance types
test_instances = [
    'ml.g4dn.xlarge',
    'ml.g4dn.2xlarge', 
    'ml.g5.xlarge',
    'ml.g5.2xlarge',
    'ml.p3.2xlarge',
]

print("\nNote: This script can't directly check instance availability,")
print("but here are some common GPU instances to try:\n")

for instance in test_instances:
    print(f"  • {instance}")

print("\n" + "="*80)
print("Checking S3 access...")
print("="*80)

# Check S3 access
s3_client = boto3.client('s3')
test_bucket = '4k-eugene-btt'

try:
    response = s3_client.head_bucket(Bucket=test_bucket)
    print(f"✓ Can access bucket: {test_bucket}")
    
    # List some objects
    response = s3_client.list_objects_v2(
        Bucket=test_bucket,
        Prefix='hdf5_data_diphone_encoded/',
        MaxKeys=5
    )
    
    if 'Contents' in response:
        print(f"✓ Found {len(response['Contents'])} objects in hdf5_data_diphone_encoded/")
        print("\nSample paths:")
        for obj in response['Contents'][:3]:
            print(f"  - {obj['Key']}")
    else:
        print("✗ No objects found in hdf5_data_diphone_encoded/")
        print("  Make sure you've uploaded the data!")
        
except Exception as e:
    print(f"✗ Error accessing S3 bucket: {e}")
    print("\nCheck:")
    print("  1. Bucket name is correct")
    print("  2. You have S3 read permissions")
    print("  3. Data has been uploaded")

print("\n" + "="*80)
print("Checking PyTorch framework availability...")
print("="*80)

try:
    # This won't catch all issues, but gives a hint
    from sagemaker.pytorch import PyTorch
    print("✓ SageMaker PyTorch SDK imported successfully")
    print("\nSupported framework versions vary by region.")
    print("Try: 2.1.0, 2.0.1, 1.13.1")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "="*80)
print("Next Steps")
print("="*80)
print("""
If you're still getting "Requested resource not found":

1. Instance Type Issues:
   Try a different instance type, e.g.:
   --instance-type ml.g4dn.xlarge
   
2. Framework Version Issues:
   Some regions don't support PyTorch 2.1.0
   Edit launch_sagemaker_job.py line 92:
   framework_version='2.0.1'  # or '1.13.1'
   
3. IAM Role Issues:
   Make sure your execution role has:
   - AmazonSageMakerFullAccess
   - S3 read access to your bucket
   
4. Regional Availability:
   Some instance types aren't available in all regions
   Common regions with good GPU support:
   - us-east-1 (N. Virginia)
   - us-west-2 (Oregon)
   - eu-west-1 (Ireland)

To see the FULL error message, run with verbose output:
Add this before estimator.fit():
    import logging
    logging.basicConfig(level=logging.DEBUG)
""")

