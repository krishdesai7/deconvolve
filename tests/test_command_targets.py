def test_primary_workflow_has_a_domain_entry_point() -> None:
    from deconvolve.workflows.train import run

    assert callable(run)


def test_leakage_check_has_a_package_entry_point() -> None:
    from deconvolve.workflows.leakage import run_leakage_check

    assert callable(run_leakage_check)


def test_batch_orchestrators_have_domain_names() -> None:
    from deconvolve.baselines.ibu import evaluate_runs as evaluate_ibu_runs
    from deconvolve.evaluation.evaluate import evaluate_runs

    assert callable(evaluate_runs)
    assert callable(evaluate_ibu_runs)


def test_the_variance_design_has_package_entry_points() -> None:
    from deconvolve.uncertainty import collect, run_cell

    assert callable(run_cell)
    assert callable(collect)


def test_the_report_has_a_package_entry_point() -> None:
    from deconvolve.reporting.report import build_report

    assert callable(build_report)
