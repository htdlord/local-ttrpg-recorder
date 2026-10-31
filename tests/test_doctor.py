from ttrpg_recorder.core.doctor import CheckResult, DoctorReport, Status, check_platform


def test_worst_status_and_ok():
    report = DoctorReport([CheckResult("a", Status.OK, ""), CheckResult("b", Status.WARN, "")])
    assert report.worst is Status.WARN and report.ok
    report.results.append(CheckResult("c", Status.FAIL, ""))
    assert not report.ok


def test_empty_report_is_ok():
    assert DoctorReport().ok


def test_platform_check():
    assert check_platform().status is Status.OK
