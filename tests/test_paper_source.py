from scripts.audit_paper_source import audit


def test_latex_source_fidelity_gates():
    status=audit()
    assert status['passed'],status['errors']
    assert len(status['equations']) == 12
    assert status['references'] == 34
    assert status['figures'] == list(range(1, 9))
