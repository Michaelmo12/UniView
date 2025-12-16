"""
Generate RSA key pair for asymmetric JWT authentication
Run this once to create private.pem and public.pem
"""
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

# Generate private key
private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
    backend=default_backend()
)

# Save private key (KEEP THIS SECRET - only auth service uses this)
private_pem = private_key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption()
)

with open('private.pem', 'wb') as f:
    f.write(private_pem)

# Generate and save public key (can be shared with all microservices)
public_key = private_key.public_key()
public_pem = public_key.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo
)

with open('public.pem', 'wb') as f:
    f.write(public_pem)

print("RSA key pair generated successfully")
print("- private.pem: Keep secret, only for database service (signs tokens)")
print("- public.pem: Share with all services (verifies tokens)")
