import sys

def check_environment(data_generation: str, take_component_id: bool, transformation: str, distinguishable: bool):
    component_selector = 'Optimal_Affected_Component_Uid' if take_component_id else 'Optimal_Affected_Component'
    if (data_generation, component_selector, transformation, distinguishable) in ENVIRONMENTS:
        i = '_id' if (take_component_id and (data_generation in ['LinearShifted', 'ARol', 'GARCH'])) else ''
        t = '_' + transformation if (data_generation in ['ARol', 'GARCH']) or distinguishable else ''
        d = '_dist' if distinguishable else ''
        filename = 'data/prepared_data/' + data_generation + i + t + d + '.csv'
        return (data_generation, component_selector, transformation, distinguishable), filename
    else:
        print('This environment does not exist. Please have a look on the "environments.py" file.')
        sys.exit(0)


ENVIRONMENTS = [('Linear', 'Optimal_Affected_Component', 'raw', False),
                ('Linear', 'Optimal_Affected_Component_Uid', 'raw', False),
                ('Linear', 'Optimal_Affected_Component', 'sqt', False),
                ('Linear', 'Optimal_Affected_Component_Uid', 'sqt', False),
                ('Linear', 'Optimal_Affected_Component', 'cube', False),
                ('Linear', 'Optimal_Affected_Component_Uid', 'cube', False),
                ('Linear', 'Optimal_Affected_Component', 'log10', False),
                ('Linear', 'Optimal_Affected_Component_Uid', 'log10', False),
                ('Linear', 'Optimal_Affected_Component', 'ln', False),
                ('Linear', 'Optimal_Affected_Component_Uid', 'ln', False),
                ('Linear', 'Optimal_Affected_Component', 'log2', False),
                ('Linear', 'Optimal_Affected_Component_Uid', 'log2', False),

                ('Saturating', 'Optimal_Affected_Component', 'raw', False),
                ('Saturating', 'Optimal_Affected_Component_Uid', 'raw', False),
                ('Saturating', 'Optimal_Affected_Component', 'sqt', False),
                ('Saturating', 'Optimal_Affected_Component_Uid', 'sqt', False),
                ('Saturating', 'Optimal_Affected_Component', 'cube', False),
                ('Saturating', 'Optimal_Affected_Component_Uid', 'cube', False),
                ('Saturating', 'Optimal_Affected_Component', 'log10', False),
                ('Saturating', 'Optimal_Affected_Component_Uid', 'log10', False),
                ('Saturating', 'Optimal_Affected_Component', 'ln', False),
                ('Saturating', 'Optimal_Affected_Component_Uid', 'ln', False),
                ('Saturating', 'Optimal_Affected_Component', 'log2', False),
                ('Saturating', 'Optimal_Affected_Component_Uid', 'log2', False),

                ('Combined', 'Optimal_Affected_Component', 'raw', False),
                ('Combined', 'Optimal_Affected_Component_Uid', 'raw', False),
                ('Combined', 'Optimal_Affected_Component', 'sqt', False),
                ('Combined', 'Optimal_Affected_Component_Uid', 'sqt', False),
                ('Combined', 'Optimal_Affected_Component', 'cube', False),
                ('Combined', 'Optimal_Affected_Component_Uid', 'cube', False),
                ('Combined', 'Optimal_Affected_Component', 'log10', False),
                ('Combined', 'Optimal_Affected_Component_Uid', 'log10', False),
                ('Combined', 'Optimal_Affected_Component', 'ln', False),
                ('Combined', 'Optimal_Affected_Component_Uid', 'ln', False),
                ('Combined', 'Optimal_Affected_Component', 'log2', False),
                ('Combined', 'Optimal_Affected_Component_Uid', 'log2', False),

                ('Discontinuous', 'Optimal_Affected_Component', 'raw', False),
                ('Discontinuous', 'Optimal_Affected_Component_Uid', 'raw', False),
                ('Discontinuous', 'Optimal_Affected_Component', 'sqt', False),
                ('Discontinuous', 'Optimal_Affected_Component_Uid', 'sqt', False),
                ('Discontinuous', 'Optimal_Affected_Component', 'cube', False),
                ('Discontinuous', 'Optimal_Affected_Component_Uid', 'cube', False),
                ('Discontinuous', 'Optimal_Affected_Component', 'log10', False),
                ('Discontinuous', 'Optimal_Affected_Component_Uid', 'log10', False),
                ('Discontinuous', 'Optimal_Affected_Component', 'ln', False),
                ('Discontinuous', 'Optimal_Affected_Component_Uid', 'ln', False),
                ('Discontinuous', 'Optimal_Affected_Component', 'log2', False),
                ('Discontinuous', 'Optimal_Affected_Component_Uid', 'log2', False),

                #--------------------    Linear Shifted Data    --------------------#

                ('LinearShifted', 'Optimal_Affected_Component_Uid', 'raw', False),
                ('LinearShifted', 'Optimal_Affected_Component_Uid', 'sqt', False),
                ('LinearShifted', 'Optimal_Affected_Component_Uid', 'cube', False),
                ('LinearShifted', 'Optimal_Affected_Component_Uid', 'log10', False),
                ('LinearShifted', 'Optimal_Affected_Component_Uid', 'ln', False),
                ('LinearShifted', 'Optimal_Affected_Component_Uid', 'log2', False),

                #--------------------    Non-stationary Data    --------------------#

                ('ARol', 'Optimal_Affected_Component_Uid', 'raw', False),
                ('ARol', 'Optimal_Affected_Component_Uid', 'sqt', False),
                ('ARol', 'Optimal_Affected_Component_Uid', 'cube', False),
                ('ARol', 'Optimal_Affected_Component_Uid', 'log10', False),
                ('ARol', 'Optimal_Affected_Component_Uid', 'ln', False),
                ('ARol', 'Optimal_Affected_Component_Uid', 'log2', False),

                ('GARCH', 'Optimal_Affected_Component_Uid', 'raw', False),
                ('GARCH', 'Optimal_Affected_Component_Uid', 'sqt', False),
                ('GARCH', 'Optimal_Affected_Component_Uid', 'cube', False),
                ('GARCH', 'Optimal_Affected_Component_Uid', 'log10', False),
                ('GARCH', 'Optimal_Affected_Component_Uid', 'ln', False),
                ('GARCH', 'Optimal_Affected_Component_Uid', 'log2', False)
                ]

