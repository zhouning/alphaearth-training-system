from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import or_
from app.db.database import get_db
from app.models.domain import Xiangzhen
import urllib.request
import json

router = APIRouter()

@router.get("/external")
def get_external_areas():
    """
    获取外部 API 提供的中国行政区信息 (仅供向下兼容或宏观展示)
    """
    try:
        url = "https://3d.img.net/orbitService/front/api/area/list?type=1"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            
            areas = data.get("data", [])
            for area in areas:
                if "boundary" in area:
                    del area["boundary"]
                    
            return {
                "status": "success",
                "count": len(areas),
                "data": areas
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/search")
def search_townships(
    keyword: str = Query(..., min_length=2, description="Search keyword for township, county or city"),
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """
    搜索本地 PostGIS 数据库中的真实乡镇级行政区边界
    """
    try:
        # Search by township or county name
        results = db.query(Xiangzhen).filter(
            or_(
                Xiangzhen.township.like(f"%{keyword}%"),
                Xiangzhen.county.like(f"%{keyword}%"),
                Xiangzhen.city.like(f"%{keyword}%")
            )
        ).limit(limit).all()
        
        # Format response
        towns = []
        for r in results:
            full_name = f"{r.province}-{r.city}-{r.county}-{r.township}"
            towns.append({
                "id": full_name,
                "province": r.province,
                "city": r.city,
                "county": r.county,
                "township": r.township,
                "full_name": full_name
            })
            
        return {
            "status": "success",
            "count": len(towns),
            "data": towns
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

