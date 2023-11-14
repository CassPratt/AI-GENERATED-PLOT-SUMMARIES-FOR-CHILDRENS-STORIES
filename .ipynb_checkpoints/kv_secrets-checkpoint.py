import os
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
import logging

# Configure the logging module
logging.basicConfig(level=logging.INFO)

keyVaultName = 'AI-GPSCS-KeyVault'
KVUri = f"https://{keyVaultName}.vault.azure.net"

credential = DefaultAzureCredential()
client = SecretClient(vault_url=KVUri, credential=credential)

def create_secret(secretName, secretValue):
    logging.info(f"Creating a secret in {keyVaultName} called '{secretName}' with the value '{secretValue}' ...")
    client.set_secret(secretName, secretValue)
    logging.info("Created secret.")

def retrieve_secret(secretName):
    logging.info(f"Retrieving your secret {secretName} from {keyVaultName}.")
    retrieved_secret = client.get_secret(secretName)
    logging.info(f"Secret retrieved.")
    return retrieved_secret.value