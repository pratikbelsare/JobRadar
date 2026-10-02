# JobRadar AWS infrastructure

The SAM template defines the processing pipeline only. The local FastAPI and Streamlit
applications are intentionally not deployed here.

## Resources

- EventBridge Scheduler invokes the coordinator on a configurable schedule.
- Coordinator Lambda selects a company batch and publishes small company IDs to SQS.
- Scanner Lambda uses the existing connector, normalization, change detection, filtering,
  and S3 snapshot abstractions.
- Analysis Lambda uses the existing Bedrock provider, analysis cache, matcher, and match
  persistence.
- Both queues have a four-attempt redrive policy and a 14-day DLQ.
- DynamoDB is on-demand; S3 is private, encrypted, versioned, and blocked from public access.

## DynamoDB access pattern

`JobRadarDataTable` uses a simple shared table:

| Entity | PK | SK | Access pattern |
| --- | --- | --- | --- |
| Job | `JOB#<id>` | `CURRENT` | Load current job |
| Job version | `JOB#<id>` | `VERSION#<number>` | Immutable version history |
| Profile/company/run | `<ENTITY>#<id>` | `RECORD` | Direct lookup |
| Analysis | `ANALYSIS#...` | `RECORD` | Exact version/model/prompt cache |
| Match | `MATCH#<profile>#<job>` | `CURRENT` | Current match for a profile/job |
| Feedback/explanation | entity-specific key | `RECORD`/`CURRENT` | Direct lookup |

`GSI1` indexes job source identity keys for source-ID and canonical-URL lookup. `GSI2`
indexes entity collections for job, company, profile, run, match, and feedback listing.
Historical job versions are written with `attribute_not_exists(PK)` so retries cannot
overwrite history. The adapters store validated model JSON in a payload attribute, keeping
serialization consistent with the local repositories.

## Prerequisites and validation

Install AWS SAM CLI and configure AWS credentials only when you are ready to validate or
deploy. No deployment is performed by this repository change.

```powershell
sam validate --template-file infrastructure/template.yaml
sam build --template-file infrastructure/template.yaml
```

Later, after reviewing parameters and IAM policies:

```powershell
sam deploy --guided --template-file .aws-sam/build/template.yaml
```

The deployment needs an AWS region, Bedrock job and embedding model IDs, and a configured
candidate profile key (the profile UUID remains an explicit field in the stored model).
Credentials must come from the standard AWS credential chain; they are never stored in this
repository.

## Recommended first deployment configuration

Use `ap-south-1` for the initial dev environment and review these parameter overrides before
deployment:

```text
EnvironmentName=dev
ScheduleExpression=rate(7 days)
ScheduleState=DISABLED
CompanyBatchSize=1
JobAnalysisModelId=apac.amazon.nova-lite-v1:0
JobAnalysisFoundationModelId=amazon.nova-lite-v1:0
EmbeddingModelId=amazon.titan-embed-text-v2:0
CandidateProfileKey=pratik
AcceptableExperienceGapYears=1
MaxStretchExperienceGapYears=2
HttpTimeoutSeconds=20
ScannerTimeoutSeconds=300
AnalysisTimeoutSeconds=300
CoordinatorMemorySize=256
WorkerMemorySize=512
```

These values are examples in `samconfig.toml.example`, not business-logic constants. Model
IDs, region, schedule, batch size, names, and timeouts remain SAM parameters.

Before deployment, verify that both configured foundation models are available in the target
region and that the account has Bedrock access:

```powershell
aws bedrock list-inference-profiles `
  --region ap-south-1 `
  --query "inferenceProfileSummaries[?inferenceProfileId=='apac.amazon.nova-lite-v1:0'].[inferenceProfileId,status]"
aws bedrock get-inference-profile `
  --inference-profile-identifier apac.amazon.nova-lite-v1:0 `
  --region ap-south-1
aws bedrock list-foundation-models `
  --region ap-south-1 `
  --query "modelSummaries[?modelId=='amazon.nova-lite-v1:0' || modelId=='amazon.titan-embed-text-v2:0'].[modelId,responseStreamingSupported]"
aws bedrock get-foundation-model `
  --model-identifier amazon.nova-lite-v1:0 `
  --region ap-south-1
aws bedrock get-foundation-model `
  --model-identifier amazon.titan-embed-text-v2:0 `
  --region ap-south-1
```

The analysis role is intentionally limited to `bedrock:InvokeModel` on the Nova APAC inference
profile, the Nova foundation model in possible destination Regions, and the Titan foundation
model in the deployment Region. Review that policy in the change set, then use a one-company
smoke run to confirm the account is enabled for both models; these checks are intentionally not
run by the normal local test suite.

## Bootstrap candidate and company data

The bootstrap CLI validates Pydantic input and uses either local JSON repositories or the
existing DynamoDB repository adapters. A profile uses a stable external key and retains its
UUID when rerun, so repeated bootstrap updates the same record instead of creating another
profile.

Local dry run:

```powershell
python -m job_intelligence.bootstrap profile `
  --profile-key pratik `
  --input examples/candidate_profile.example.json `
  --backend json `
  --directory data/profiles

python -m job_intelligence.bootstrap companies `
  examples/companies.example.json `
  --backend json `
  --directory data/companies.json
```

After deployment, use the CloudFormation table output and the same reviewed input files:

```powershell
aws cloudformation describe-stacks `
  --stack-name jobradar-dev `
  --region ap-south-1 `
  --query "Stacks[0].Outputs"

python -m job_intelligence.bootstrap profile `
  --profile-key pratik `
  --input examples/candidate_profile.example.json `
  --backend dynamodb `
  --region ap-south-1 `
  --table <DataTableName>

python -m job_intelligence.bootstrap companies `
  examples/companies-smoke.json `
  --backend dynamodb `
  --region ap-south-1 `
  --table <DataTableName>
```

`companies-smoke.json` should contain one reviewed company with `enabled: true`. The checked
in example keeps the company disabled to prevent accidental crawling.

## Exact first-deployment workflow

1. Authenticate with the AWS CLI using a named profile; do not put credentials in files.
2. Select the profile and region:

   ```powershell
   $env:AWS_PROFILE = "<your-profile>"
   $env:AWS_DEFAULT_REGION = "ap-south-1"
   aws sts get-caller-identity
   ```

3. Validate and build without deploying:

   ```powershell
   sam validate --lint --template-file infrastructure/template.yaml
   sam build --template-file infrastructure/template.yaml
   ```

4. Review the generated change set and deploy manually:

   ```powershell
   sam deploy --guided --template-file .aws-sam/build/template.yaml
   ```

   Confirm the stack name, region, model IDs, `CompanyBatchSize=1`, and
   `ScheduleState=DISABLED` before accepting the change set.
5. Bootstrap the candidate profile, then bootstrap exactly one enabled smoke-test company.
6. Manually invoke the coordinator instead of waiting for EventBridge:

   ```powershell
   aws lambda invoke `
     --function-name jobradar-dev-coordinator `
     --region ap-south-1 `
     --payload '{}' `
     --cli-binary-format raw-in-base64-out `
     coordinator-response.json
   Get-Content coordinator-response.json
   ```

7. Inspect each stage:

   ```powershell
   aws logs tail /aws/lambda/jobradar-dev-coordinator --since 10m --region ap-south-1
   aws logs tail /aws/lambda/jobradar-dev-scanner --since 10m --region ap-south-1
   aws logs tail /aws/lambda/jobradar-dev-analysis --since 10m --region ap-south-1
   aws s3api list-objects-v2 --bucket <RawBucketName> --prefix raw/ --region ap-south-1
   aws dynamodb scan --table-name <DataTableName> --region ap-south-1 --select COUNT
   aws sqs get-queue-attributes --queue-url <CompanyScanQueueUrl> --region ap-south-1 --attribute-names ApproximateNumberOfMessages ApproximateNumberOfMessagesNotVisible
   aws sqs get-queue-attributes --queue-url <JobAnalysisQueueUrl> --region ap-south-1 --attribute-names ApproximateNumberOfMessages ApproximateNumberOfMessagesNotVisible
   ```

   Confirm the raw snapshot, job/version, analysis, and match records exist and both DLQs
   are empty. Check the CloudWatch logs for connector, filter, Bedrock, or persistence errors.
8. Only after the one-company path succeeds should the schedule be changed to `ENABLED` and
   the company batch size or company list increased.

The expected first smoke-test path is:

```text
Coordinator → company-scan SQS → Scanner → S3 snapshot → DynamoDB job/version
→ deterministic filter → job-analysis SQS → Nova Lite extraction
→ Titan V2 embedding → deterministic ranking → DynamoDB match
```

## Cost notes

DynamoDB on-demand, Lambda, SQS, Scheduler, and S3 are usage-based. Bedrock inference is
the largest variable cost, so unchanged jobs stay out of the analysis queue and analysis is
cached by job version, model ID, and prompt version. S3 storage and noncurrent versions can
grow over time; the template expires noncurrent object versions after 30 days.
