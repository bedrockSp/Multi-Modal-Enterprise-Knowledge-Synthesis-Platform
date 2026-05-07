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


# DocumentGraph metadata — body of the graph lives in Kuzu/NetworkX on disk;
# this collection only tracks status + counts for fast listing.
document_graph_schema = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["user_id", "thread_id", "status"],
        "properties": {
            "user_id": {"bsonType": "string"},
            "thread_id": {"bsonType": "string"},
            "status": {"enum": ["pending", "building", "ready", "failed"]},
            "node_count": {"bsonType": ["int", "long"]},
            "edge_count": {"bsonType": ["int", "long"]},
            "community_count": {"bsonType": ["int", "long"]},
            "doc_count": {"bsonType": ["int", "long"]},
            "built_at": {"bsonType": ["date", "null"]},
            "created_at": {"bsonType": "date"},
            "updated_at": {"bsonType": "date"},
            "error": {"bsonType": "string"},
        },
    }
}


try:
    db.create_collection("document_graphs", validator=document_graph_schema)
    db.document_graphs.create_index([("user_id", 1), ("thread_id", 1)], unique=True)
    print("Collection 'document_graphs' created with schema validation.")
except CollectionInvalid:
    print("Collection 'document_graphs' already exists.")
except Exception as e:
    print("Error creating document_graphs collection:", e)
