from fastapi import Query
from app.schemas import BaseResponse
from app.services import QdrantService

class QdrantController:
    def __init__(self, bot_service: QdrantService):
        self.bot_service = bot_service
        self.qdrant_service = QdrantService()

    async def dump_data_to_qdrant(self) -> BaseResponse:
        """API endpoint to dump all Jira and Confluence data to Qdrant"""
        try:
            result = self.qdrant_service.dump_all_data_to_qdrant()
            
            return BaseResponse(
                message=result["message"],
                data=result
            )
            
        except Exception as e:
            return BaseResponse(
                message="Error dumping data to Qdrant",
                data={"error": str(e)}
            )

    async def get_qdrant_stats(self) -> BaseResponse:
        """Get Qdrant collection statistics"""
        try:
            collection_info = self.qdrant_service.qdrant_client.get_collection(
                self.qdrant_service.collection_name
            )
            
            return BaseResponse(
                message="Qdrant statistics retrieved",
                data={
                    "collection_name": self.qdrant_service.collection_name,
                    "points_count": collection_info.points_count,
                    "vectors_count": collection_info.vectors_count,
                    "status": collection_info.status
                }
            )
            
        except Exception as e:
            return BaseResponse(
                message="Error getting Qdrant stats",
                data={"error": str(e)}
            )
