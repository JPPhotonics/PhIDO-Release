from MyHelpers import *

# get a list of directories to process
dir_list = [
    "./nodes/mmi1x2/",
]


for node_dir in dir_list:
    current_timestamp = timestamp()
    sys.stdout = MyLogger(node_dir + f"{current_timestamp}_log.txt")

    with open(node_dir + "define.yaml") as file:
        node_def = yaml.safe_load(file)

    print()
    pprint(node_def)
    print()

    optimize_param, x0 = parse_opt_params(node_def)
    print(x0)
    print(optimize_param)

    ##########################

    parameter_space = []
    for i in range(len(optimize_param)):
        # parameter_space.append( Real(low=optimize_param[i][1], high=optimize_param[i][2], name=optimize_param[i][0]) )
        # parameter_space.append( Integer(low=optimize_param[i][1], high=optimize_param[i][2], name=optimize_param[i][0]) )
        parameter_space.append(
            Categorical(optimize_param[i][1], name=optimize_param[i][0])
        )

    # Create a new function that has some parameters fixed
    partial_objective = partial(
        objective_wrapper, yamlDefinition=node_def
    )  # can pass more than one constant

    # Running the optimization
    result = gp_minimize(
        func=partial_objective,  # the objective function to minimize
        dimensions=parameter_space,  # the range of parameters
        n_calls=30,  # the number of evaluations of `func`
        # acq_func="LCB", #  gp_hedge (default), LCB, EI, PI
        # n_initial_points = 10, # default: 10
        # n_random_starts=5,             # the number of random initialization points
        # noise=0.1**2,                  # the noise level (optional)
        # random_state=42                # the random seed
        x0=x0,
    )

    # result = gbrt_minimize(func=partial_objective, dimensions=parameter_space, n_calls=20, x0 = x0)

    # with open(node_dir+f'optimization_result_{current_timestamp}.pickle', 'wb') as f:
    #     pickle.dump(result, f)
    dump(result, node_dir + f"{current_timestamp}_OptimizeResultObj.pickle")
    dump(dict(result), node_dir + f"{current_timestamp}_optimization_result.pickle")
    dump_(sim_results, node_dir + f"{current_timestamp}_sim_result.pickle")

    # Optimal parameters
    pprint(node_def["yamlCode"])
    print("============")
    print("Optimal parameters:", result.x)

    # Minimum value of the objective function
    print("Minimum value of the objective:", result.fun)

    # All evaluated points and corresponding function values
    print("\nAll tested parameters and function values:")
    for x, f_val in zip(result.x_iters, result.func_vals):
        print(f"Parameters: {x}, Objective: {f_val}")

    # Total number of evaluations of the objective function
    print("\nTotal evaluations:", len(result.x_iters))

    # Details of the optimization algorithm (e.g., GP kernel, acquisition function)
    print("\nOptimizer details:")
    print("GP Kernel:", result.specs["args"]["base_estimator"])
