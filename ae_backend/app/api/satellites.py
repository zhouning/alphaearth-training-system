from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.models.domain import SmSatellite
import urllib.request
import json

router = APIRouter()

@router.get("/external")
def get_external_satellites():
    """
    获取外部 API 关注的卫星数据源
    """
    try:
        url = "https://3d.img.net/orbitService/front/api/satellite/list?resolution=&country=&satelliteType="
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            return {
                "status": "success",
                "count": len(data.get("data", [])),
                "data": data.get("data", [])
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/available")
def get_available_satellites(db: Session = Depends(get_db)):
    """
    获取系统中所有可用的卫星数据源 (直接从旧版 sm_satellite 表中读取)
    """
    try:
        # Fetch satellites where norad_number is not null
        satellites = db.query(SmSatellite).filter(SmSatellite.norad_number.isnot(None)).all()
        
        result = []
        for sat in satellites:
            result.append({
                "norad_number": sat.norad_number,
                "name": sat.data_source,
                "type": sat.type
            })
            
        return {
            "status": "success",
            "count": len(result),
            "data": result
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
