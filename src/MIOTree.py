import numpy as np
import pyomo.environ as pyo
from scipy import stats
from src import Tree as CustomTree
from sklearn import tree
from sklearn.metrics import accuracy_score
import time
import sys


class MIOTree:

    def __init__(self, alpha, min_samples_per_leaf, max_depth, X_train, y_train, for_tuning=False):
        self.alpha = alpha
        self.min_samples_per_leaf = min_samples_per_leaf
        self.max_depth = max_depth
        self.X_train = X_train
        self.y_train = y_train
        self.tree = CustomTree.Tree(max_depth)

        self.create_model(for_tuning)

    def solve(self, solver_name='glpk', warmstart=False):
        """
        Solve the Pyomo model.
        Tested solvers: 
            - glpk: Not passed
            - cbc: ?
            - gurobi: passed
            - cplex: ?
            - ipopt: passed
        """
        solver = pyo.SolverFactory(solver_name)
        results = None
        try :
            results = solver.solve(self.pyomo_model, warmstart=warmstart)
        except:
            results = solver.solve(self.pyomo_model)
        return results
    
    def print_log(self, model_type="MIO", duration=None, accuracy_train=None, accuracy_test=None, depth=None, alpha=None):
        """
        Print the log of the model.
        """
        if depth is None:
            depth = self.max_depth
        if alpha is None:
            alpha = self.alpha
        if duration is None:
            duration = "-"
        if accuracy_train is None:
            accuracy_train = "-"
        if accuracy_test is None:
            accuracy_test = "-"
        print(f"{model_type}\tDepth: {depth}\tAlpha: {alpha}\tDuration: {duration}")
        print(f"\tAccuracy train: {accuracy_train}\tAccuracy test: {accuracy_test}")

    def cart_model(self, X_train, y_train, D, max_leaf_nodes=None, min_samples_per_leaf=1, alpha=0):
        cart_model = None
        if self.min_samples_per_leaf > 1:
            cart_model = tree.DecisionTreeClassifier(max_depth=D, min_samples_leaf=min_samples_per_leaf, ccp_alpha=alpha, max_leaf_nodes=max_leaf_nodes)
        else:
            cart_model = tree.DecisionTreeClassifier(max_depth=D, ccp_alpha=alpha, max_leaf_nodes=max_leaf_nodes)
        
        init_time = time.time()
        cart_model = cart_model.fit(X_train, y_train)
        duration = time.time() - init_time

        return cart_model, duration

    def apply_cart_to_mio(self, tree_from_model):
        # create a map of the nodes
        node_map = {1:0}
        for t in self.pyomo_model.branch_nodes:
            left = 2 * t
            right = 2 * t + 1

            node_map[left] = -1
            node_map[right] = -1
            
            left_child = tree_from_model.children_left[node_map[t]]
            node_map[left] = left_child
            
            right_child = tree_from_model.children_right[node_map[t]]
            node_map[right] = right_child

        # create nodes
        nodes = {t: self.create_node(tree_from_model, node_map[t]) for t in self.pyomo_model.branch_nodes}
        nodes.update({t: self.create_node(tree_from_model, node_map[t], is_leaf=True) for t in self.pyomo_model.leaf_nodes})

        features_indices_len = len(self.pyomo_model.features_indices)

        # Set the branch nodes
        for t in self.tree.branch_nodes:
            feature = nodes[t]["feature"]
            threshold = nodes[t]["threshold"]

            if feature is None or feature == -2: # model.tree_.TREE_UNDEFINED = -2
                self.pyomo_model.d[t] = 0
                self.pyomo_model.b[t] = None
                for f in range(features_indices_len):
                    self.pyomo_model.a[f, t] = 0
            else:
                feature = int(feature)
                self.pyomo_model.d[t] = 1
                self.pyomo_model.b[t] = threshold
                for f in range(features_indices_len):
                    self.pyomo_model.a[f, t] = 1 if f == feature else 0

        # Set the leaf nodes
        """ for t in self.tree.leaf_nodes:
            value = nodes[t]["value"]
            if value is None:
                self.pyomo_model.l[t] = int(t % 2)
                if t % 2:
                    while nodes[t]["value"] is None:
                        t //= 2
                    for k in self.pyomo_model.classes_indices:
                        self.pyomo_model.c[k, t] = 1 if k == np.argmax(value) else 0
                else:
                    for k in self.pyomo_model.classes_indices:
                        self.pyomo_model.c[k, t] = 0
            else:
                self.pyomo_model.l[t] = 1
                for k in self.pyomo_model.classes_indices:
                    if k == np.argmax(value):
                        self.pyomo_model.c[k, t] = 1
                    else:
                        self.pyomo_model.c[k, t] = 0 """
            
    def tune(self, X_test, y_test):
        """
        Tune the hyperparameters of the model.
        :param X_test: np.ndarray
        :param y_test: np.ndarray
        """
        warm_start_pool = []

        for D in range(2, self.max_depth + 1):
            C_max = np.power(2, D) - 1
            for C in range(1, C_max + 1):
                # Run CART using Nmin with α = 0. Trim the solution to depth D and to a maximum of C splits.
                # !!! It's not possible to set max_samples_split in DecisionTreeClassifier !!!
                
                cart_model = None
                max_leaf_nodes = C + 1 # == max_samples_split
                cart_model, duration = self.cart_model(self.X_train, self.y_train, D, max_leaf_nodes=max_leaf_nodes, min_samples_per_leaf=self.min_samples_per_leaf, alpha=0)

                tree_from_model = cart_model.tree_

                new_model = MIOTree(
                    alpha=C,
                    max_depth=D,
                    min_samples_per_leaf=self.min_samples_per_leaf,
                    X_train=self.X_train,
                    y_train=self.y_train,
                    for_tuning=True
                )

                new_model.apply_cart_to_mio(tree_from_model)

                accuracy = accuracy_score(self.y_train, cart_model.predict(self.X_train))
                self.print_log("CART", duration=duration, accuracy_train=accuracy, depth=D, alpha=C)

                # Search through pool of candidate warm starts (including CART) and choose the one with the lowest error
                new_warm_start_pool = np.concatenate((warm_start_pool, [{'model': new_model, 'accuracy': accuracy, 'duration': duration, 'type': 'CART'}]))
                best_warm_start = max(new_warm_start_pool, key=lambda x: x['accuracy'])
                print(f"Best warm-start: {best_warm_start}")
                best_model = best_warm_start['model']
                
                # Solve MIO problem for depth D and C splits using the selected warm start
                new_MIO_model = MIOTree(
                    alpha=C,
                    max_depth=D,
                    min_samples_per_leaf=self.min_samples_per_leaf,
                    X_train=self.X_train,
                    y_train=self.y_train,
                    for_tuning=True
                )                    

                features_indices_len = len(new_model.pyomo_model.features_indices)
                for t in best_model.tree.branch_nodes:
                    # if exists else None
                    new_MIO_model.pyomo_model.d[t] = best_model.pyomo_model.d[t]
                    #new_MIO_model.pyomo_model.b[t] = best_model.pyomo_model.b[t]
                    for f in range(features_indices_len):
                        new_MIO_model.pyomo_model.a[f, t] = best_model.pyomo_model.a[f, t]

                """ for t in best_model.tree.leaf_nodes:
                    new_MIO_model.pyomo_model.l[t] = best_model.pyomo_model.l[t]
                    for k in best_model.pyomo_model.classes_indices:
                        new_MIO_model.pyomo_model.c[k, t] = best_model.pyomo_model.c[k, t] """
            
                init_time = time.time()
                new_MIO_model.solve('gurobi', warmstart=True)
                duration = time.time() - init_time

                accuracy = new_MIO_model.calculate_accuracy()
                self.print_log(duration=duration, accuracy_train=accuracy, depth=D, alpha=C)
                leaf_predictions = self.extract_leaf_predictions(new_MIO_model)
                self.tree.print_tree(leaf_predictions, a=new_MIO_model.pyomo_model.a, b=new_MIO_model.pyomo_model.b)
                print()

                # Add the MIO solution to the warm start pool
                warm_start_pool = np.concatenate((warm_start_pool, [{'model': new_MIO_model, 'accuracy': accuracy, 'duration': duration, 'type': 'MIO'}]))

        # Post-process the solution pool to remove all solutions that are not optimal to (24) for any value of α
        print()
        print("Post-processing the solution pool")
        optimal_solutions = []
        for i in range(len(warm_start_pool)):
            model = warm_start_pool[i]['model']
            accuracy = model.calculate_accuracy()
            warm_start_pool[i]['accuracy'] = accuracy

            # Remove constraint
            model.pyomo_model.del_component(model.pyomo_model.constraint_C)
            model.pyomo_model.del_component(model.pyomo_model.objective_function)

            # Add new objective function
            model.pyomo_model.objective_function = pyo.Objective(rule=model.objective_function_no_tuning, sense=pyo.minimize)

            init_time = time.time()
            result = model.solve('gurobi', warmstart=True)
            duration = time.time() - init_time
            accuracy_test = model.calculate_accuracy(X_test, y_test)
            self.print_log("MIO", duration=duration, accuracy_train=accuracy, depth=model.max_depth, alpha=model.alpha, accuracy_test=accuracy_test)

            if result.solver.termination_condition == pyo.TerminationCondition.optimal:
                optimal_solutions.append(warm_start_pool[i])
            else:
                print(f"Solution is not optimal for alpha: {model.alpha}")

        # Identify the best performing solution on a validation set, and the range of α for which
        # this solution is optimal. Use the midpoint of this interval as the tuned value of α
        # If same accuracy, choose the one with the lowest duration
        best_result = max(optimal_solutions, key=lambda x: (x['model'].calculate_accuracy(X_test, y_test), -x['duration']))
        final_model = best_result['model']

        print(f"Final model accuracy: {best_result['accuracy']}, alpha: {final_model.alpha}, depth: {final_model.max_depth}")
        
        return final_model
    
    def extract_leaf_predictions(self, model = None):
        if model is None:
            model = self

        classes = np.unique(model.y_train)

        num_leaf_nodes = len(model.pyomo_model.leaf_nodes)
        leaf_predictions = [None] * num_leaf_nodes

        class_index_to_label = {i: classes[i] for i in model.pyomo_model.classes_indices}

        for i in model.pyomo_model.classes_indices:
            for j in model.pyomo_model.leaf_nodes:
                if int(model.pyomo_model.c[i, j].value) == 1:
                    leaf_index = j - num_leaf_nodes
                    leaf_predictions[leaf_index] = class_index_to_label[i]

        return leaf_predictions

    def calculate_confusion_matrix(self):
        y = self.y_train.flatten() if len(self.y_train.shape) > 1 else self.y_train
        
        if self.y_pred is None:
            self.y_pred = self.extract_predictions()
            if len(self.y_pred.shape) > 1:
                self.y_pred = self.y_pred.flatten()
        
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)

        for i in range(n_classes):
            for j in range(n_classes):
                confusion_matrix[i, j] = np.sum((y == unique_classes[i]) & (self.y_pred == unique_classes[j]))

        return confusion_matrix

    def calculate_accuracy(self, X_test=None, y_test=None):
        """
        Calculate the accuracy of the model.
        :param X_test: np.ndarray
        :param y_test: np.ndarray
        """
        y = y_test if y_test is not None else self.y_train
        if len(y.shape) > 1:
            y = y.flatten()

        self.y_pred = self.extract_predictions(X_test)
        if len(self.y_pred.shape) > 1:
            self.y_pred = self.y_pred.flatten()
        return np.sum(self.y_pred == y) / len(y)

    def extract_predictions(self, X_test=None):
        """
        Extract the predictions from the Pyomo model.
        """
        unique_classes = np.unique(self.y_train)
        labels = {t: k for t in self.pyomo_model.leaf_nodes for k in unique_classes if self.pyomo_model.c[k, t].value == 1}
        X_set = X_test if X_test is not None else self.X_train
        predictions = []
        features_indices = list(self.pyomo_model.features_indices)

        for x in X_set:
            t = 1
            while t not in self.pyomo_model.leaf_nodes:
                a_t = [self.pyomo_model.a[j, t]() for j in features_indices]
                b_t = self.pyomo_model.b[t]()
                
                sum_term = np.sum(a_t[j] * x[j] for j in range(len(features_indices)))

                if sum_term >= b_t: # go right
                    t = 2 * t + 1
                else: # go left
                    t = 2 * t
            predictions.append(labels[t])

        return np.array(predictions)

    def print_model(self):
        """
        Print the Pyomo model.
        """
        if self.pyomo_model is not None:
            self.pyomo_model.pprint()
        else:
            print("Model creation failed. Model is None.")
    
    def create_model(self, for_tuning) -> pyo.ConcreteModel:
        """
        Create the Pyomo model for the decision tree.
        """
        model = pyo.ConcreteModel("MIOTree")

        # compute epsilon value (uj)
        epsilons, epsilon_min, epsilon_max = self.compute_epsilon_min_max(self.X_train)

        # one-hot encode the target variable
        y_one_hot = self.one_hot_encode(self.y_train)

        # calculate the baseline accuracy (baseline_accuracy)
        baseline_accuracy = self.calculate_baseline_accuracy(self.y_train)

        # M constant same as the number of training samples
        n_constant = len(self.X_train)

        # build arrays
        data_indices = range(self.X_train.shape[0]) # called 'n' in paper
        model.data_indices = pyo.Set(initialize=data_indices, ordered=True, name="data_indices", doc="datapoints index")
        
        features_indices = range(self.X_train.shape[1])  # called 'p' in paper
        model.features_indices = pyo.Set(initialize=features_indices, ordered=True, name="features_indices", doc="features index")
       
        classes_indices = np.unique(self.y_train) # called 'K' in paper
        model.classes_indices = pyo.Set(initialize=classes_indices, ordered=True, name="classes_indices", doc="classes index")

        branch_nodes = self.tree.branch_nodes # called 'TB' in paper
        model.branch_nodes = pyo.Set(initialize=branch_nodes, ordered=True, name="branch_nodes", doc="branch nodes")
        leaf_nodes = self.tree.leaf_nodes # called 'TL' in paper
        model.leaf_nodes = pyo.Set(initialize=leaf_nodes, ordered=True, name="leaf_nodes", doc="leaf nodes")

        # add variables
        model.a = pyo.Var(features_indices, branch_nodes, domain=pyo.Binary, name="a", doc="split parameter (transposed(a) * xi < b)")
        model.b = pyo.Var(branch_nodes, domain=pyo.Reals, name="b", doc="split threshold (transposed(a) * xi < b)")
        model.c = pyo.Var(classes_indices, leaf_nodes, domain=pyo.Binary, name="c", doc="node prediction")
        model.d = pyo.Var(branch_nodes, domain=pyo.Binary, name="d", doc="is branch active")
        model.l = pyo.Var(leaf_nodes, domain=pyo.Binary, name="l", doc="leaf node activation")
        model.L = pyo.Var(leaf_nodes, domain=pyo.NonNegativeIntegers, name="L", doc="leaf node misclassified")
        model.z = pyo.Var(data_indices, leaf_nodes, domain=pyo.Binary, name="z", doc="leaf node assignment")
        model.Nt = pyo.Var(leaf_nodes, domain=pyo.NonNegativeIntegers, name="Nt", doc="number of samples in leaf node")
        model.Nkt = pyo.Var(classes_indices, leaf_nodes, domain=pyo.NonNegativeIntegers, name="Nkt", doc="number of samples in leaf node with class k")

        # add parameters
        model.x = pyo.Param(data_indices, features_indices, initialize=dict(np.ndenumerate(self.X_train)), within=pyo.Any, name="x", doc="datapoints")
        model.y = pyo.Param(data_indices, initialize=dict(enumerate(self.y_train)), within=pyo.Any, name="y", doc="target variable")
        model.y_one_hot = pyo.Param(data_indices, classes_indices, initialize=dict(np.ndenumerate(y_one_hot)), within=pyo.Any, name="y_one_hot", doc="one-hot encoded target variable")
        model.epsilons = pyo.Param(features_indices, initialize=dict(enumerate(epsilons)), within=pyo.Any, name="epsilons", doc="epsilon values")
        model.epsilon_max = pyo.Param(initialize=epsilon_max, name="epsilon_max", doc="maximum epsilon value")
        model.epsilon_min = pyo.Param(initialize=epsilon_min, name="epsilon_min", doc="minimum epsilon value")
        model.baseline_accuracy = pyo.Param(initialize=baseline_accuracy, name="baseline_accuracy", doc="baseline accuracy")
        model.n_constant = pyo.Param(initialize=n_constant, name="n_constant", doc="constant value")
        model.alpha = pyo.Param(initialize=self.alpha, name="alpha", doc="regularization parameter")
        model.min_samples_per_leaf = pyo.Param(initialize=self.min_samples_per_leaf, name="min_samples_per_leaf", doc="minimum number of samples per leaf")
        model.max_depth = pyo.Param(initialize=self.max_depth, name="max_depth", doc="max_depth of the tree")

        # add objective function
        
        if for_tuning:
            model.objective_function = pyo.Objective(rule=self.objective_function_tuning, sense=pyo.minimize)
        else:
            model.objective_function = pyo.Objective(rule=self.objective_function_no_tuning, sense=pyo.minimize)

        # add constraints
        if for_tuning: # alpha == C
            model.constraint_C = pyo.Constraint(rule=lambda m: sum(m.d[t] for t in m.branch_nodes) <= m.alpha)

        model.constraint_01 = pyo.Constraint(branch_nodes, rule=lambda m, t: sum(m.a[j, t] for j in m.features_indices) == m.d[t])
        model.constraint_02_a = pyo.Constraint(branch_nodes, rule=lambda m, t: m.b[t] <= m.d[t])
        model.constraint_02_b = pyo.Constraint(branch_nodes, rule=lambda m, t: m.b[t] >= 0)

        def constraint_03_a(m, j, t): # ok, is Binary
            return (m.a[j, t] == 0 or m.a[j, t] == 1) and (m.d[t] == 0 or m.d[t] == 1)
        #model.constraint_03_a = pyo.Constraint(features_indices, branch_nodes, rule=constraint_03_a)

        def constraint_03_b(m, i, t): # ok, is Binary
            return (m.z[i, t] == 0 or m.z[i, t] == 1) and (m.l[t] == 0 or m.l[t] == 1)
        #model.constraint_03_b = pyo.Constraint(data_indices, leaf_nodes, rule=constraint_03_b)

        model.constraint_04 = pyo.Constraint(branch_nodes[1:], rule=lambda m, t: m.d[t] <= m.d[t//2]) # t//2 is the parent node
        model.constraint_05 = pyo.Constraint(data_indices, leaf_nodes, rule=lambda m, i, t: m.z[i, t] <= m.l[t])
        model.constraint_06 = pyo.Constraint(leaf_nodes, rule=lambda m, t: sum(m.z[i, t] for i in m.data_indices) >= m.min_samples_per_leaf * m.l[t])
        model.constraint_07 = pyo.Constraint(data_indices, rule=lambda m, i: sum(m.z[i, t] for t in m.leaf_nodes) == 1)

        def constraint_08(_m, m, t, i): # ok
            e_min = _m.epsilon_min
            return (pyo.quicksum(_m.a[j, m] * (_m.x[i, j] + _m.epsilons[j] - e_min) for j in _m.features_indices) + e_min) <= _m.b[m] + (1 + _m.epsilon_max) * (1 - _m.z[i, t])
        left_ancestors_per_leaf = [self.tree.left_ancestors(i) for i in leaf_nodes]
        data = self.calculate_combination_arrays(data_indices, leaf_nodes, left_ancestors_per_leaf)
        model.constraint_08 = pyo.Constraint(data, rule=constraint_08)

        def constraint_09(_m, m, t, i): # ok
            return pyo.quicksum(_m.a[j, m] * _m.x[i, j] for j in _m.features_indices) >= _m.b[m] - (1 - _m.z[i, t])
        right_ancestors_per_leaf = [self.tree.right_ancestors(i) for i in leaf_nodes]
        data = self.calculate_combination_arrays(data_indices, leaf_nodes, right_ancestors_per_leaf)
        model.constraint_09 = pyo.Constraint(data, rule=constraint_09)

        def constraint_10(m, t, k): # ok
            return m.Nkt[k, t] == 0.5 * pyo.quicksum(m.z[i, t] * (1 + m.y_one_hot[i, k]) for i in m.data_indices)                  
        model.constraint_10 = pyo.Constraint(leaf_nodes, classes_indices, rule=constraint_10)

        model.constraint_11 = pyo.Constraint(leaf_nodes, rule=lambda m, t: sum(m.z[i, t] for i in m.data_indices) == m.Nt[t])
        model.constraint_12 = pyo.Constraint(leaf_nodes, rule=lambda m, t: sum(m.c[k, t] for k in m.classes_indices) == m.l[t])
        model.constraint_13 = pyo.Constraint(leaf_nodes, classes_indices, rule=lambda m, t, k: m.L[t] >= m.Nt[t] - m.Nkt[k, t] - m.n_constant * (1 - m.c[k, t]))
        model.constraint_14 = pyo.Constraint(leaf_nodes, classes_indices, rule=lambda m, t, k: m.L[t] <= m.Nt[t] - m.Nkt[k, t] + m.n_constant * model.c[k, t])
        model.constraint_15 = pyo.Constraint(leaf_nodes, rule=lambda m, t: m.L[t] >= 0)

        def constraint_16(m): # always true
            return m.C == m.d.sum()
        #model.constraint_16 = pyo.Constraint(rule=constraint_16)

        self.pyomo_model = model

    def objective_function_no_tuning(self, m):
        return (sum(m.L[t] for t in m.leaf_nodes) / m.baseline_accuracy) + m.alpha * sum(m.d[t] for t in m.branch_nodes)
        
    def objective_function_tuning(self, m):
        return (sum(m.L[t] for t in m.leaf_nodes) / m.baseline_accuracy) #+ m.alpha * sum(m.d[t] for t in m.branch_nodes)

    def create_node(self, tree_model, index, is_leaf=False):
        if index == -1:
            return {'feature': None, 'threshold': None, 'value': None}
        
        return {
            'feature': None if is_leaf else tree_model.feature[index],
            'threshold': None if is_leaf else tree_model.threshold[index],
            'value': tree_model.value[index, 0]
        }

    def one_hot_encode(self, y):
        """
        One-hot encode the target variable.
        """
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.flatten()

        unique_classes, y_indices = np.unique(y, return_inverse=True)
        n_samples = len(y)
        n_classes = len(unique_classes)
        # fill of -1
        Y = np.full((n_samples, n_classes), -1, dtype=int)
        Y[np.arange(n_samples), y_indices] = 1
        return Y

    def calculate_baseline_accuracy(self, y):
        """
        Calculate the baseline accuracy for the target variable.
        """
        classes = np.unique(y)
        max = 0
        for c in classes:
            count = len(y[y == c])
            if count > max:
                max = count
        return max/len(y)

    def compute_epsilon_min_max(self, X):
        """"
        Compute the minimum and maximum epsilon values for each feature in the dataset.
        """
        eps = []
        for j in range(X.shape[1]):
            values = np.unique(X[:, j])
            values = np.sort(values)
            values = np.diff(values)
            values = values[values != 0]  # remove 0 values
            if len(values) > 0:
                eps.append(np.min(values))
            else:
                eps.append(0)

        return eps, np.min(eps), np.max(eps)
    
    def calculate_combination_arrays(self, data_indices, leaf_nodes, ancestors_per_leaf):
        """
        Calculate the combination arrays for the constraints.
        """
        result = []
        for i in data_indices:
            for j in range(len(leaf_nodes)):
                for x in ancestors_per_leaf[j]:
                    result.append((int(x), int(leaf_nodes[j]), int(i)))

        return result