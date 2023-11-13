import os
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

os.environ['KEY_VAULT_NAME'] = 'AI-GPSCS-KeyVault'

keyVaultName = os.environ['KEY_VAULT_NAME']
KVUri = f"https://{keyVaultName}.vault.azure.net"

credential = DefaultAzureCredential()
client = SecretClient(vault_url=KVUri, credential=credential)

def create_secret(secretName, secretValue):
    print(f"Creating a secret in {keyVaultName} called '{secretName}' with the value '{secretValue}' ...")
    client.set_secret(secretName, secretValue)
    print("Created secret.")

def retrieve_secret(secretName):
    print(f"Retrieving your secret {secretName} from {keyVaultName}.")
    retrieved_secret = client.get_secret(secretName)
    print(f"Secret retrieved.")
    return retrieved_secret.value