"""MongoDB connection testing and encoding utility."""
import os
import sys
import socket
import ssl
import requests
from pathlib import Path
from urllib.parse import quote_plus

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from pipelines.mongodb_store import MongoDBStore


def encode_mongodb_uri(uri: str) -> str:
    """
    Encode special characters in MongoDB connection string password.
    
    Args:
        uri: MongoDB connection string
        
    Returns:
        Encoded connection string
    """
    if not uri.startswith('mongodb'):
        return uri
    
    if '://' not in uri:
        return uri
    
    scheme_part = uri.split('://', 1)
    scheme = scheme_part[0]
    rest = scheme_part[1]
    
    parts = rest.split('@')
    
    if len(parts) < 2:
        return uri
    
    hostname_part = parts[-1]
    credentials = '@'.join(parts[:-1])
    
    if ':' not in credentials:
        return uri
    
    username = credentials.split(':', 1)[0]
    password = credentials.split(':', 1)[1]
    
    encoded_password = quote_plus(password)
    encoded_uri = f"{scheme}://{username}:{encoded_password}@{hostname_part}"
    
    return encoded_uri


def test_basic_connectivity(hostname: str, port: int = 27017) -> bool:
    """Test basic TCP connectivity to MongoDB."""
    print(f"Testing TCP connectivity to {hostname}:{port}...")
    try:
        sock = socket.create_connection((hostname, port), timeout=10)
        sock.close()
        print(f"  SUCCESS: Can reach {hostname}:{port}")
        return True
    except Exception as e:
        print(f"  FAILED: Cannot reach {hostname}:{port}")
        print(f"  Error: {str(e)}")
        return False


def test_ssl_connection(hostname: str) -> bool:
    """Test SSL connection to MongoDB."""
    print(f"Testing SSL connection to {hostname}...")
    try:
        context = ssl.create_default_context()
        sock = socket.create_connection((hostname, 27017), timeout=10)
        ssock = context.wrap_socket(sock, server_hostname=hostname)
        ssock.close()
        print(f"  SUCCESS: SSL handshake successful")
        return True
    except Exception as e:
        print(f"  FAILED: SSL handshake failed")
        print(f"  Error: {str(e)}")
        return False


def test_mongodb_connection_direct(uri: str, options: dict = None) -> bool:
    """Test MongoDB connection directly with pymongo."""
    if options is None:
        options = {
            'serverSelectionTimeoutMS': 30000,
            'connectTimeoutMS': 30000,
            'socketTimeoutMS': 30000,
            'retryWrites': True,
            'retryReads': True,
        }
    
    print("Testing MongoDB connection...")
    try:
        client = MongoClient(uri, **options)
        result = client.admin.command('ping')
        server_info = client.server_info()
        print(f"  SUCCESS: Connected to MongoDB")
        print(f"  MongoDB version: {server_info.get('version', 'Unknown')}")
        client.close()
        return True
    except ServerSelectionTimeoutError as e:
        print(f"  FAILED: Server selection timeout")
        print(f"  Error: {str(e)[:200]}")
        return False
    except ConnectionFailure as e:
        print(f"  FAILED: Connection failure")
        print(f"  Error: {str(e)[:200]}")
        return False
    except Exception as e:
        print(f"  FAILED: Unexpected error")
        print(f"  Error: {str(e)[:200]}")
        return False


def test_connection_via_store() -> bool:
    """Test MongoDB connection via MongoDBStore."""
    print("Testing MongoDB connection via MongoDBStore...")
    try:
        store = MongoDBStore()
        print(f"  SUCCESS: Connected to MongoDB")
        print(f"  Database: {store.database_name}")
        print(f"  Features collection: {store.features_collection_name}")
        print(f"  Models collection: {store.models_collection_name}")
        store.close()
        return True
    except Exception as e:
        error_msg = str(e)
        print(f"  FAILED: Connection error")
        print(f"  Error: {error_msg[:200]}...")
        return False


def extract_hostname(uri: str) -> str:
    """Extract hostname from MongoDB connection string."""
    if '@' in uri:
        hostname_part = uri.split('@')[1].split('/')[0].split('?')[0]
        return hostname_part.split(':')[0] if ':' in hostname_part else hostname_part
    return ""


def print_encoding_help():
    """Print encoding help information."""
    print("\nSpecial characters that need URL-encoding:")
    print("  @ -> %40")
    print("  # -> %23")
    print("  / -> %2F")
    print("  : -> %3A")
    print("  ? -> %3F")
    print("  & -> %26")
    print("  = -> %3D")
    print("  + -> %2B")


def get_current_ip():
    """Get current public IP address."""
    try:
        response = requests.get('https://api.ipify.org', timeout=5)
        if response.status_code == 200:
            return response.text.strip()
    except:
        pass
    return None

def print_troubleshooting():
    """Print troubleshooting steps."""
    current_ip = get_current_ip()
    
    print("\nTroubleshooting Steps:")
    print("1. URL-encode special characters in password")
    print("2. Verify MongoDB Atlas Network Access:")
    print("   - Go to MongoDB Atlas -> Network Access")
    if current_ip:
        print(f"   - Your current IP appears to be: {current_ip}")
        print(f"   - Ensure this IP (or 0.0.0.0/0 for all IPs) is whitelisted")
    else:
        print("   - Add your IP address (or 0.0.0.0/0 for all IPs)")
    print("   - Wait 1-2 minutes after adding IP for changes to propagate")
    print("3. Verify database user permissions:")
    print("   - Go to MongoDB Atlas -> Database Access")
    print("   - Ensure user has 'readWrite' permissions")
    print("4. Check connection string format:")
    print("   mongodb+srv://username:encoded_password@cluster.mongodb.net/database_name?retryWrites=true&w=majority")
    print("5. For SSL/TLS errors:")
    print("   - Verify your IP is whitelisted in MongoDB Atlas")
    print("   - Check Windows Firewall/Antivirus settings")
    print("   - Try temporarily disabling firewall to test")
    print("   - Try connecting from MongoDB Compass to verify credentials")
    print("   - If using corporate network, check proxy settings")


def main():
    """Main function."""
    load_dotenv()
    
    # Handle command-line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == 'encode':
            if len(sys.argv) < 3:
                print("Usage: python test_mongodb.py encode <connection_string>")
                print("\nExample:")
                print('python test_mongodb.py encode "mongodb+srv://user:pass@456#@cluster.mongodb.net/"')
                print_encoding_help()
                sys.exit(1)
            
            uri = sys.argv[2]
            print(f"Original URI: {uri}")
            encoded_uri = encode_mongodb_uri(uri)
            
            if encoded_uri != uri:
                print(f"\nEncoded URI: {encoded_uri}")
                print("\nCopy this to your .env file:")
                print(f"MONGODB_URI={encoded_uri}")
            else:
                print("\nNo encoding needed or URI format not recognized")
            return
        
        elif sys.argv[1] == 'diagnose':
            mongodb_uri = os.getenv('MONGODB_URI')
            if not mongodb_uri:
                print("ERROR: MONGODB_URI not found in .env file")
                sys.exit(1)
            
            print("=" * 60)
            print("MongoDB Connection Diagnostics")
            print("=" * 60)
            print(f"\nConnection string (masked): {mongodb_uri.split('@')[0] if '@' in mongodb_uri else mongodb_uri[:50]}...")
            
            hostname = extract_hostname(mongodb_uri)
            if not hostname:
                print("ERROR: Could not extract hostname from connection string")
                sys.exit(1)
            
            print(f"\nExtracted hostname: {hostname}\n")
            
            test_basic_connectivity(hostname, 27017)
            test_ssl_connection(hostname)
            
            print("\n" + "=" * 60)
            print("Testing MongoDB Connection")
            print("=" * 60)
            test_mongodb_connection_direct(mongodb_uri)
            
            print("\n" + "=" * 60)
            print("Diagnostics Complete")
            print("=" * 60)
            print_troubleshooting()
            return
    
    # Default: Test connection
    mongodb_uri = os.getenv('MONGODB_URI')
    
    if not mongodb_uri:
        print("ERROR: MONGODB_URI not found in .env file")
        print("\nPlease add your MongoDB connection string to .env file:")
        print("MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/database_name?retryWrites=true&w=majority")
        print("\nTo encode a connection string with special characters:")
        print('python test_mongodb.py encode "mongodb+srv://user:pass@123#@cluster.mongodb.net/"')
        print_encoding_help()
        sys.exit(1)
    
    print("=" * 60)
    print("MongoDB Connection Test")
    print("=" * 60)
    print(f"\nConnection string (masked): {mongodb_uri.split('@')[0] if '@' in mongodb_uri else mongodb_uri[:50]}...")
    
    # Check for encoding issues
    if '@' in mongodb_uri.split('://')[1].split('@')[0] if '://' in mongodb_uri else '':
        print("\nWARNING: Password may contain unencoded '@' character")
        print("Use 'python test_mongodb.py encode <connection_string>' to encode it")
    
    print("\n" + "-" * 60)
    success = test_connection_via_store()
    print("-" * 60)
    
    if not success:
        print_troubleshooting()
        print("\nFor detailed diagnostics, run:")
        print("python test_mongodb.py diagnose")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Connection test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
