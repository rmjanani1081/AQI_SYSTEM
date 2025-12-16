import hashlib
from datetime import datetime

def create_audit_hash(payload: dict):
    record = str(payload) + str(datetime.utcnow())
    return hashlib.sha256(record.encode()).hexdigest()
