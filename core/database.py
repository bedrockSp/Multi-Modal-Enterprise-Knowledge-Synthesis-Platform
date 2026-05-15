from pymongo import MongoClient
from pymongo.errors import CollectionInvalid

from core.config import settings

MONGO_URI = settings.DATABASE_URL
client = MongoClient(MONGO_URI)
db = client[settings.DATABASE_NAME]


user_schema = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["userId", "name", "email", "password", "is_active", "threads"],
        "properties": {
            "userId": {"bsonType": "string"},
            "name": {"bsonType": "string"},
            "email": {"bsonType": "string"},
            "password": {"bsonType": "string"},
            "is_active": {"bsonType": "bool"},
            "threads": {
                "bsonType": "object",
                "additionalProperties": {
                    "bsonType": "object",
                    "required": [
                        "documents",
                        "chats",
                        "createdAt",
                        "updatedAt",
                        "extra_done",
                        "mindmap_enabled",
                    ],
                    "properties": {
                        "thread_name": {"bsonType": "string"},
                        "documents": {
                            "bsonType": "array",
                            "items": {
                                "bsonType": "object",
                                "required": [
                                    "docId",
                                    "title",
                                    "type",
                                    "time_uploaded",
                                    "file_name",
                                ],
                                "properties": {
                                    "docId": {"bsonType": "string"},
                                    "title": {"bsonType": "string"},
                                    "type": {"bsonType": "string"},
                                    "file_name": {"bsonType": "string"},
                                    "time_uploaded": {"bsonType": "date"},
                                },
                            },
                        },
                        "chats": {
                            "bsonType": "array",
                            "items": {
                                "bsonType": "object",
                                "required": ["type", "content", "timestamp"],
                                "properties": {
                                    "type": {"enum": ["agent", "user"]},
                                    "content": {"bsonType": "string"},
                                    "timestamp": {"bsonType": "date"},
                                    "sources": {
                                        "bsonType": "object",
                                        "properties": {
                                            "documents_used": {
                                                "bsonType": "array",
                                                "items": {
                                                    "bsonType": "object",
                                                    "required": [
                                                        "document_id",
                                                    ],
                                                    "properties": {
                                                        "title": {"bsonType": "string"},
                                                        "document_id": {
                                                            "bsonType": "string"
                                                        },
                                                        "page_no": {
                                                            "bsonType": ["int", "long"]
                                                        },
                                                    },
                                                },
                                            },
                                            "web_used": {
                                                "bsonType": "array",
                                                "items": {
                                                    "bsonType": "object",
                                                    "properties": {
                                                        "title": {
                                                            "bsonType": [
                                                                "string",
                                                                "null",
                                                            ]
                                                        },
                                                        "url": {
                                                            "bsonType": [
                                                                "string",
                                                                "null",
                                                            ]
                                                        },
                                                        "favicon": {
                                                            "bsonType": [
                                                                "string",
                                                                "null",
                                                            ]
                                                        },
                                                    },
                                                },
                                            },
                                        },
                                    },
                                },
                            },
                        },
                        "createdAt": {"bsonType": "date"},
                        "updatedAt": {"bsonType": "date"},
                        "extra_done": {
                            "bsonType": "bool",
                            "description": "Indicates if extra task is done",
                        },
                        "mindmap_enabled": {
                            "bsonType": "bool",
                            "description": "Mindmap feature enabled for this thread",
                        },
                        "instructions": {
                            "bsonType": "array",
                            "description": "User-defined instructions for this thread",
                            "items": {
                                "bsonType": "object",
                                "required": ["id", "text", "selected"],
                                "properties": {
                                    "id": {"bsonType": "string"},
                                    "text": {"bsonType": "string"},
                                    "selected": {"bsonType": "bool"},
                                },
                            },
                        },
                    },
                },
            },
        },
    }
}


try:
    db.create_collection("users", validator=user_schema)
    db.users.create_index("userId", unique=True)
    print("Collection 'users' created with schema validation.")
except CollectionInvalid:
    print("Collection 'users' already exists.")
except Exception as e:
    print("Error creating collection:", e)


