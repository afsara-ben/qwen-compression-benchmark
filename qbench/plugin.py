def register():
    import os
    if os.environ.get('MOECBENCH_MARLIN_CANONICAL_ORDER') == '1':
        from runtime.marlin_order import install
        install()
