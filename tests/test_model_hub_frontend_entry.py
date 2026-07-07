from pathlib import Path


FRONTEND = Path(__file__).resolve().parents[1] / "ae_frontend" / "index.html"


def _read_frontend_html() -> str:
    return FRONTEND.read_text(encoding="utf-8")


def _read_frontend_template() -> str:
    html = _read_frontend_html()
    return html[html.find("<body") : html.rfind("<script>")]


def test_frontend_exposes_model_hub_tab_and_api_actions():
    html = _read_frontend_html()

    assert "currentTab = 'modelHub'" in html
    assert "currentTab === 'modelHub'" in html
    assert "模型中心" in html
    assert "/api/ae/model-hub/models" in html
    assert "/api/ae/model-hub/jobs" in html
    assert "runModelHubDemo" in html
    assert "modelHubModels" in html


def test_frontend_uses_model_hub_runtime_modes_for_demo_jobs():
    html = _read_frontend_html()

    assert "getModelHubDemoInputMode" in html
    assert "package_profile?.runtime_modes" in html
    assert "default_demo_input_mode" in html
    assert "model.task_type === 'change_detection'" in html
    assert "model.status === 'demo_only'" in html
    assert "model.model_id === 'lulc_6class_prithvi_houlsby'" in html
    assert "prithvi_crop_classification_arcgis_style" not in html


def test_frontend_exposes_metadata_driven_crop_raster_demo_controls():
    html = _read_frontend_html()

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
    html = _read_frontend_html()

    assert "systemCapabilities" in html
    assert "fetchSystemCapabilities" in html
    assert "/api/ae/system/capabilities" in html
    assert "systemCapabilities" in html
    assert "AlphaEarth 系统" in html
    assert "readiness_counts" in html
    assert "evidence_sources" in html
    assert "arcgis_replacement" in html
    assert "Paper12 能力总览" not in html


def test_frontend_exposes_system_capability_card_hooks():
    html = _read_frontend_html()

    assert "systemCapabilityFor" in html
    assert "capabilityWorkflowLabel" in html
    assert "formatEvidenceValue" in html
    assert "systemCapabilityFor(model).workflow_level" in html
    assert "systemCapabilityFor(model).checkpoint" in html
    assert "systemCapabilityFor(model).limitations" in html
    assert "systemCapabilityFor(model).next_steps" in html

def test_frontend_exposes_model_hub_job_summary_sections():
    html = _read_frontend_html()

    assert "summarizeModelHubJob" in html
    assert "任务摘要" in html
    assert "输出制品" in html
    assert "运行日志" in html
    assert "原始 JSON" in html
    assert "modelHubJob.artifacts" in html
    assert "modelHubJob.logs" in html


def test_frontend_exposes_system_verification_workbench_hooks():
    html = _read_frontend_html()

    assert "/api/ae/system/verification" in html
    assert "systemVerification" in html
    assert "loadingSystemVerification" in html
    assert "systemVerificationError" in html
    assert "fetchSystemVerification" in html
    assert "verificationForCapability" in html
    assert "verificationStatusClass" in html
    assert "overall_status" in html
    assert "next_actions" in html
    assert "checks" in html
    assert "systemVerificationRawJson" in html


def test_frontend_keeps_model_hub_job_controls_with_system_verification():
    html = _read_frontend_html()

    assert "runModelHubDemo" in html
    assert "runModelHubRasterDemo" in html
    assert "modelHubJob.artifacts" in html
    assert "modelHubJob.logs" in html
    assert "fetchSystemCapabilities" in html
    assert "fetchSystemVerification" in html


def test_frontend_exposes_system_evidence_drilldown_hooks():
    html = _read_frontend_html()
    template = _read_frontend_template()

    assert "/api/ae/system/evidence" in html
    assert "systemEvidence" in html
    assert "loadingSystemEvidence" in html
    assert "systemEvidenceError" in html
    assert "fetchSystemEvidence" in html
    assert "证据下钻" in template
    assert "证据制品预览" in template
    assert "Evidence drill-down" not in template
    assert "Artifact previews" not in template
    assert "evidenceForCheck" in html
    assert "evidenceStatusClass" in html
    assert "formatArtifactSize" in html
    assert "formatArtifactPreview" in html
    assert "evidenceStatusLabel" in html
    assert "artifactKindLabel" in html
    assert "localizeSystemMessage" in html
    assert "预览" in template



def test_frontend_localizes_visible_model_hub_status_labels():
    html = _read_frontend_html()
    template = _read_frontend_template()

    assert "statusLabel" in html
    assert "taskTypeLabel" in html
    assert "inputModeLabel" in html
    assert "verificationStatusLabel" in html
    assert "evidenceStatusLabel" in html
    assert "artifactKindLabel" in html
    assert "replacementStatusLabel" in html
    assert "localizeSystemMessage" in html
    assert "verificationForCapability(model).next_actions" in html
    assert "系统能力自检" in template
    assert "systemVerification.overall_status" in html
    assert "verificationStatusLabel(systemVerification.overall_status)" in html
    assert "阻塞问题" in template
    assert "verificationForCapability(model).next_actions" in html
    assert "System verification" not in template
    assert "Blocking issues" not in template
    assert "next actions" not in template.lower()


def test_frontend_localizes_visible_lulc_labels():
    template = _read_frontend_template()

    assert "产品来源" in template
    assert "年份" in template
    assert "标签掩膜" in template
    assert "掩膜尺寸" in template
    assert "confidence_summary" in template
    assert "pixel_accuracy" in template
    assert "前期（2025Q1） / 后期（2025Q4） / RGB 差异" in template
    assert ">Provider</span>" not in template
    assert ">Year</span>" not in template
    assert "Label mask" not in template
    assert "mask shape" not in template
    assert "mean confidence" not in template
    assert "Pixel accuracy:" not in template
    assert "Before (2025Q1) / After (2025Q4) / RGB Diff" not in template


def test_frontend_localizes_chart_tooltips_and_runtime_messages():
    html = _read_frontend_html()

    assert "可训练参数量" in html
    assert "[演示]" in html
    assert "128x128 切片" in html
    assert "随机种子 #" in html
    assert "参数 =" in html
    assert "35,920 张切片数据集" in html
    assert "const params = new URLSearchParams" in html
    assert "Trainable params" not in html
    assert "<br/>params =" not in html
    assert "epoch/run" not in html
    assert "seed #" not in html
    assert "35,920 patches" not in html
    assert "35,920 patch" not in html


def test_frontend_keeps_model_hub_controls_with_system_evidence():
    html = _read_frontend_html()

    assert "runModelHubDemo" in html
    assert "runModelHubRasterDemo" in html
    assert "fetchSystemCapabilities" in html
    assert "fetchSystemVerification" in html
    assert "fetchSystemEvidence" in html
    assert "systemVerificationRawJson" in html
    assert "modelHubJob.artifacts" in html
    assert "modelHubJob.logs" in html

def test_frontend_exposes_production_readiness_labels():
    html = _read_frontend_html()

    assert "productionStateLabel" in html
    assert "runtimeKindLabel" in html
    assert "生产候选" in html
    assert "需要下载" in html
    assert "需要训练" in html
    assert "真实推理" in html
    assert "契约演示" in html


def test_frontend_warns_when_model_uses_demo_or_cached_outputs():
    html = _read_frontend_html()

    assert "不要将演示结果解释为生产推理" in html
