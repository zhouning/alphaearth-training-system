from pathlib import Path


FRONTEND = Path(__file__).resolve().parents[1] / "ae_frontend" / "index.html"


def test_frontend_exposes_model_hub_tab_and_api_actions():
    html = FRONTEND.read_text(encoding="utf-8")

    assert "currentTab = 'modelHub'" in html
    assert "currentTab === 'modelHub'" in html
    assert "妯″瀷涓績" in html
    assert "/api/ae/model-hub/models" in html
    assert "/api/ae/model-hub/jobs" in html
    assert "runModelHubDemo" in html
    assert "modelHubModels" in html


def test_frontend_uses_model_hub_runtime_modes_for_demo_jobs():
    html = FRONTEND.read_text(encoding="utf-8")

    assert "getModelHubDemoInputMode" in html
    assert "package_profile?.runtime_modes" in html
    assert "default_demo_input_mode" in html
    assert "model.task_type === 'change_detection'" in html
    assert "model.status === 'demo_only'" in html
    assert "model.model_id === 'lulc_6class_prithvi_houlsby'" in html
    assert "prithvi_crop_classification_arcgis_style" not in html


def test_frontend_exposes_metadata_driven_crop_raster_demo_controls():
    html = FRONTEND.read_text(encoding="utf-8")

    assert "upload_raster_demo" in html
    assert "supportsModelHubRasterDemo" in html
    assert "modelHubRasterPath" in html
    assert "runModelHubRasterDemo" in html
    assert "supportsModelHubRasterDemo(model)" in html
    assert "runtimeModes.includes('upload_raster_demo')" in html
    assert 'v-if="supportsModelHubRasterDemo(model)"' in html
    assert "isModelHubExecutable" in html
    assert "isModelHubRunnable(model) || supportsModelHubRasterDemo(model)" in html
    assert "modelHubJob.value.status === 'failed'" in html
    assert "modelHubJob.value.error" in html
    assert "HTTP ${res.status}" in html
    assert "res.statusText" in html
    assert "prithvi_crop_inputs" in html
    assert "D:/tmp/crop_18band.tif" not in html


def test_frontend_exposes_system_capability_workbench():
    html = FRONTEND.read_text(encoding="utf-8")

    assert "systemCapabilities" in html
    assert "fetchSystemCapabilities" in html
    assert "/api/ae/system/capabilities" in html
    assert "系统能力工作台" in html
    assert "AlphaEarth System" in html
    assert "readiness_counts" in html
    assert "evidence_sources" in html
    assert "arcgis_replacement" in html
    assert "Paper12 能力总览" not in html


def test_frontend_exposes_system_capability_card_hooks():
    html = FRONTEND.read_text(encoding="utf-8")

    assert "systemCapabilityFor" in html
    assert "capabilityWorkflowLabel" in html
    assert "formatEvidenceValue" in html
    assert "systemCapabilityFor(model).workflow_level" in html
    assert "systemCapabilityFor(model).checkpoint" in html
    assert "systemCapabilityFor(model).limitations" in html
    assert "systemCapabilityFor(model).next_steps" in html

def test_frontend_exposes_model_hub_job_summary_sections():
    html = FRONTEND.read_text(encoding="utf-8")

    assert "summarizeModelHubJob" in html
    assert "任务摘要" in html
    assert "输出制品" in html
    assert "运行日志" in html
    assert "原始 JSON" in html
    assert "modelHubJob.artifacts" in html
    assert "modelHubJob.logs" in html
