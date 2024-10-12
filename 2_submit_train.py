# %%
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
import os
from azure.ai.ml import command
from azure.ai.ml.entities import JobResourceConfiguration

# %%
if os.path.exists("configs.env"):
    load_dotenv("configs.env")

# %%
# authenticate
credential = DefaultAzureCredential()

# %%
SUBSCRIPTION = os.getenv("SUBSCRIPTION_ID")
RESOURCE_GROUP = os.getenv("RESOURCE_GROUP")
WS_NAME = os.getenv("WORKSPACE_NAME")

# %%
# Get a handle to the workspace
ml_client = MLClient(
    credential=credential,
    subscription_id=SUBSCRIPTION,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WS_NAME,
)

# %%
# Verify that the handle works correctly.
ws = ml_client.workspaces.get(WS_NAME)
print(ws.location, ":", ws.resource_group)

# %%
job = command(
    code="verify_install",  # location of source code
    command="python tester.py", # script to run
    environment="mmrotate-azure-env@latest",
    display_name="mmrotate-verify-install",
    resources=JobResourceConfiguration(
        instance_type="Standard_NC8as_T4_v3", instance_count=1
    ),
    queue_settings={"job_tier": "spot"},
)

# %%
# submit the job
ml_client.create_or_update(job)

# %%
