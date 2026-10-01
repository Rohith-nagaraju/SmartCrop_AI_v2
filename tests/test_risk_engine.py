from app.services.risk_engine import dpi_from_environment,fuse_risk

def test_dpi_range(): assert 0 <= dpi_from_environment(25,70,5,55) <= 100

def test_fusion_range():
    score,level=fuse_risk(.8,10,60); assert 0<=score<=100; assert level in {'LOW','MODERATE','HIGH','CRITICAL'}
