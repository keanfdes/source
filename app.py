from flask import Flask, render_template, request
import numpy as np
import pandas as pd

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Retrieve input data from the form
        A = np.array([[float(x) for x in row.split(',')] for row in request.form['A'].split('\n')])
        print(A)
        equalities = np.array(request.form['equalities'].split(','))
        print(type(equalities[1]))
        b = np.array([float(x) for x in request.form['b'].split(',')])
        c = np.array([float(x) for x in request.form['c'].split(',')])

        # Call the simplex method function with the input data
        result = simplex_method(A, equalities, b, c)

        return render_template('index.html', result=result)
    return render_template('index.html')

def simplex_method(A, equalities, b, c):
    np.set_printoptions(precision=6, suppress=True)
    eqs = {"less than":"<=", "equal":"=", "more than":">="}
    tol = 1e-15

    # This function prints the objective function and constraints in readable LP form for better understanding.
    # It takes a parameter phase where it prints the objective function with artificial variables when phase = 1 
    # and normal c array coefficients when it has any other value.
    def printLP(X, phase):
        # print objective function
        if phase == 1:
            obj_function_1 = -np.ones(num_artificial_variables)
            variables_str = "+".join(cost_variables[1 + num_variables + num_slack_variables : 1 + num_variables + num_slack_variables + num_artificial_variables])
            result = "min " + variables_str
        else:
            obj_function = "Minimize Z = " + " + ".join([f"{c[i]}x{i+1}" for i in range(len(c))])
            result = obj_function
        result += "<br>Subject to:"
        # Print the constraints
        for i in range(X.shape[0]):
            constraint = " + ".join([f"{X[i][j]}x{j+1}" for j in range(X.shape[1])])
            result += f"<br>Constraint {i+1}: {constraint} {eqs[equalities[i]]} {b[i]}"
        return result

    # Combines the tableau matrix(new_array) along with the basis array on the left-hand side 
    # and constraint, slack and artificial name (cost variable) array on the top.
    # This is just for display purposes, no operations are run on this.
    def print_tableau():
        # Initializing by combining new_array with cost_variables array on top as columns
        df_new_array = pd.DataFrame(new_array, columns=cost_variables)
        # Creating a new DataFrame for basis and transposing it
        df_basis = pd.DataFrame([basis]).T
        # Concatenating the basis DataFrame and the new_array DataFrame horizontally
        df_final = pd.concat([df_basis, df_new_array], axis=1)
        # Round the numbers up to a decimal place of 6 while displaying very small numbers as 0 and ignoring non-numerical values
        def round_if_numeric(x, tol):
            if isinstance(x, (int, float)):
                return round(x, 6) if abs(x) > tol else 0
            return x
        df_rounded = df_final.applymap(lambda x: round_if_numeric(x, tol))
        return df_rounded.to_html(index=False)

    # Function that takes A and b as input and outputs the reduced row echelon form of the concatenated matrix of A & B.
    # Python does not have a function like rref for this purpose so this and only this function was taken from an article
    # http://www.ryanhmckenna.com/2021/03/removing-redundant-constraints-from.html and modified by me to suit my execution.
    def reduced_row_echelon(A, b):
        # Tolerance to avoid division by numbers close to zero
        tol = 1e-15
        # To avoid mismatch if A or b consists of integers and operations need to be performed with floats
        A = A.astype(float)
        b = b.astype(float)
        # Concatenate A and b
        Ab = np.concatenate((A, b.reshape(-1, 1)), axis=1)
        m, n = Ab.shape
        r = 0  # Row counter
        for c in range(n):
            if r == m:
                break
            # Find the pivot row
            pivot = np.argmax(np.abs(Ab[r:m, c])) + r
            max_val = np.abs(Ab[pivot, c])
            if max_val <= tol:
                # Skip column c, making sure the pivot is below tol
                Ab[r:m, c] = 0
                continue
            # Swap the current row with the pivot row
            Ab[[r, pivot], c:n] = Ab[[pivot, r], c:n]
            # Normalize the pivot row
            Ab[r, c:n] = Ab[r, c:n] / Ab[r, c]

            # Eliminate the current column
            v = Ab[r, c:n]
            if r > 0:
                Ab[:r, c:n] = Ab[:r, c:n] - np.outer(Ab[:r, c], v)
            if r < m-1:
                Ab[r+1:m, c:n] = Ab[r+1:m, c:n] - np.outer(Ab[r+1:m, c], v)
            r += 1  # Move to the next row
        # Make the pivot columns identity
        for i in range(min(m, n)):
            Ab[i, :] /= Ab[i, i] if np.abs(Ab[i, i]) > tol else 1

        return Ab

    # Function to check if any RHS elements are negative, it then performs operations so that RHS becomes positive
    def negative_RHS():
        nonlocal A, b, equalities
        indices_non_positive = np.array([])
        # Adding to an array all indices where RHS is negative
        if(np.any(b <= 0)):
            indices_non_positive = np.where(b <= 0)[0]
        # For each index, perform a set of operations to make RHS positive
        for indice in indices_non_positive:
            # Find negative of LHS
            A[indice] = -A[indice]
            # Negate RHS to make it positive
            b[indice] = -b[indice]
            # Switch equalities to opposite
            if(equalities[indice] == "less than"):
                equalities[indice] = "more than"
            elif(equalities[indice] == "more than"):
                equalities[indice] = "less than"

    # This function obtains the reduced row echelon form of the combined constraint matrix, checks if there are any rows with all zeros in them,
    # then deletes those rows from the A, b and equalities matrices which basically removes the redundant constraints from the process.
    def redundancy_checker():
        nonlocal A, b, equalities
        # Obtain reduced row echelon form
        A_reduced = reduced_row_echelon(A, b)
        # Indices of rows with all zeros
        zero_rows = np.where(~A_reduced.any(axis=1))[0]
        # Deleting matching rows from A, b and equalities matrices
        A = np.delete(A, zero_rows, axis=0)
        b = np.delete(b, zero_rows, axis=0)
        equalities = np.delete(equalities, zero_rows, axis=0)

    # Defining Bland's rule to prevent cycling for both columns (finding max cost) and rows (finding minimum ratio)
    def blands_rule(extrema, minimum_ratios=None):
        result = ""
        if extrema == "max":
            # Finding max value column in the first row, excluding the first and last elements
            max_value = np.max(np.round(new_array[0][1:-1], 3))
            # Creating an array with indices in which the maximum value occurs, if there are no repeating values, array has only 1 value
            indices = np.where(np.round(new_array[0][1:-1], 3) == max_value)[0]
            # If there are repeating values, print the indices of the occurrences along with the values
            if len(indices) > 1:
                result = f"Since columns {indices + 1} have the same cost {new_array[0][indices[0]+1]}, we take the one with lowest index according to Bland's rule which is column {indices[0]+1}"
            # Chosen pivot is the first pivot at index 0 / minimum index with 1 added to it to account for z row
            pivot = 1 + indices[0]
        elif extrema == "min":
            # Finding minimum ratio from minimum ratio array by taking the minimum positive value
            min_ratio = min(ratio for ratio in minimum_ratios[1:] if ratio > 0)
            # Finding indices of repeating values in minimum ratio array
            indices = [i for i, ratio in enumerate(minimum_ratios[1:], start=1) if ratio == min_ratio]
            # Chosen pivot is the first pivot at index 0 / minimum index
            if len(indices) > 1:
                result = f"Since rows {indices + 1} have the same minimum ratio {minimum_ratios[indices[0]]}, we take the one with lowest index according to Bland's rule which is row {indices[0]}"
            pivot = indices[0]
        return pivot, result

    # Performs tableau iterations by finding pivot row and column, dividing so pivot row & column intersection is 1 
    # and rest of the rows in the same column is 1 using row operations.
    def tableau_iter():
        iter_no = 1
        result = ""
        # Checking if all values in the tableau reduced cost row are less than 0, if so the loop terminates, else it continues
        while(np.any(np.round(new_array[0][1:-1], 2) > 0)):
            result += f"<br><br>Iteration {iter_no}"
            # Finding pivot column using Bland's Rule
            pivot_column, bland_result = blands_rule("max")
            result += f"<br>{bland_result}"
            # Checking if all values in the pivot column are negative, if so the problem is unbounded and the program terminates
            if np.all(new_array[:, pivot_column]) < 0:
                result += "<br>The problem is unbounded and hence there is no solution"
                return result
            # Initializing empty minimum ratios array
            minimum_ratios = []
            # Filling the array with values of RHS(last column) divided by corresponding pivot column value
            for i in new_array:
                minimum_ratios.append(i[-1]/i[pivot_column])
            result += "<br>minimum ratios: " + str(minimum_ratios[1:])
            # Finding pivot row by finding least positive minimum ratio
            pivot_row, bland_result = blands_rule("min", minimum_ratios)
            result += f"<br>pivot row (leaving element) = {pivot_row}"
            result += f"<br>pivot column (entering element) = {pivot_column}"
            # Updating the basis by replacing leaving element with entering element
            basis[pivot_row] = cost_variables[pivot_column]
            result += f"<br>pivot element = {new_array[pivot_row][pivot_column]}"

            # Dividing pivot row by pivot element to make pivot element 1
            new_array[pivot_row] = new_array[pivot_row]/new_array[pivot_row][pivot_column]
            # Performing row operations to make the rest of the elements in the pivot column 0
            for row_index, row in enumerate(new_array):
                if row_index != pivot_row:
                    new_array[row_index][1:] -= (new_array[pivot_row][1:] * row[pivot_column])
            result += print_tableau()
            iter_no += 1
        return result

    result = ""
    # Original LP
    result += "Our original LP<br>"
    result += printLP(A, 0)
    # Checking for redundancy and updating the LP
    redundancy_checker()
    # Display after removing redundant constraints
    result += "<br><br>After removing redundant constraints<br>"
    result += printLP(A, 0)
    # Checking for negative RHS
    negative_RHS()
    # Displaying after accounting for negative variables in RHS
    result += "<br><br>After accounting for negative variables in RHS<br>"
    result += printLP(A, 0)
    # Counting number of problem variables, slack variables and artificial variables
    num_variables = A.shape[1]
    num_slack_variables = np.sum((equalities == "less than") | (equalities == "more than"))
    num_artificial_variables = np.sum((equalities == "equal") | (equalities == "more than"))
    # Create augmented matrix which includes space for slack and artificial variables
    aug_A = np.zeros((A.shape[0], A.shape[1] + num_slack_variables + num_artificial_variables))
    # Copy the original data without slack and artificial coefficients to the matrix
    aug_A[:, :A.shape[1]] = A
    result += "<br><br>Adding slack variables....."
    # Index where the first slack variable column starts (which is the end of the original constraints matrix)
    slack_column_index = A.shape[1]
    flag = 0
    for eq_index, eq_value in enumerate(equalities):
        if eq_value == "less than":
            # Adding slack variable if equality is less than
            aug_A[eq_index, slack_column_index] = 1
            # Move to the next slack variable column
            slack_column_index += 1
            flag = 1
        elif eq_value == "more than":
            # Subtracting slack variable if equality is more than
            aug_A[eq_index, slack_column_index] = -1
            slack_column_index += 1
            flag = 1
    # Checking if any slack variables have been added or not and giving appropriate response
    if flag == 0:
        result += "<br>There are no slack variables"
    else:
        result += "<br><br>After adding slack variables<br>"
        result += printLP(aug_A, 0)
    result += "<br><br>Adding artificial variables......"
    # Index where the first artificial variable column starts (which is the length of the augmented matrix with all variables included minus the number of artificial variables)
    artificial_column_index = aug_A.shape[1] - num_artificial_variables
    flag = 0
    for eq_index, eq_value in enumerate(equalities):
        if eq_value == "more than":
            # Adding artificial variable if equality is more than
            aug_A[eq_index, slack_column_index] = 1
            # Move to the next slack variable column
            slack_column_index += 1
            flag = 1
        elif eq_value == "equal":
            # Adding artificial variable if equality is equality
            aug_A[eq_index, slack_column_index] = 1
            slack_column_index += 1
            flag = 1
    # Checking if any artificial variables have been added or not and giving appropriate response
    if flag == 0:
        result += "<br>There are no artificial variables"
    else:
        result += "<br><br>After adding artificial variables<br>"
        result += printLP(aug_A, 0)
    # Initializing first phase
    result += "<br><br>First phase"
    # Making cost index column names
    cost_variables = ["z"]
    for i in range(num_variables + num_slack_variables + num_artificial_variables):
        cost_variables.append(f"x{i+1}")
    cost_variables.append("RHS")
    # Creating basis, for each row in aug_A the code searches from the end of the row to find the first non-zero element,
    # indicating a basic variable. Once found, the corresponding variable name from cost_variables is added to the basis list.
    basis = ["z"]
    for row in aug_A:
        for i in range(len(row) - 1, -1, -1):
            if row[i] != 0:
                # Found the first non-zero element, print its index
                basis.append(cost_variables[i+1])
                break
    result += printLP(aug_A, 1)
    
    # Creating tableau matrix filled with zeros with dimensions accounting for cost(Zj-Cj) row and z and RHS columns in addition to augmented A matrix
    new_array = np.zeros((aug_A.shape[0] + 1, aug_A.shape[1] + 2))

    # Adding augmented matrix to the tableau
    new_array[1:1+aug_A.shape[0], 1:-1] = aug_A

    # Adding b matrix to the tableau
    new_array[1:1+aug_A.shape[0], -1] = b

    # Keeping top left as 1
    new_array[0, 0] = 1

    # For phase 1, populating the 0th row(cost) with -1's for every artificial variable
    new_array[0, 1+artificial_column_index:1+artificial_column_index+num_artificial_variables] = -np.ones(num_artificial_variables)

    result += "<br><br>Basis: " + str(basis)
    result += "<br>Zj-Cj: " + str(cost_variables)
    result += "<br><br>Initial tableau<br>"
    result += print_tableau()

    # Preparing the array for 1st phase by performing row operations to make the artificial variable costs to 0
    for row in new_array[1:]:
        if(np.any(row[1+artificial_column_index:1+artificial_column_index+num_artificial_variables] > 0)):
            new_array[0] += row

    result += "<br><br>After preparing for phase 1<br>"
    result += print_tableau()

    # Perform 1st phase tableau operations
    result += tableau_iter()

    # Terminating the program if RHS cost is ~0 and declaring the problem as infeasible
    if new_array[0][-1] < -tol or new_array[0][-1] > tol:
        result += "<br>The problem is infeasible"
        return result

    # Getting the values of artificial variables
    artificial_vars = cost_variables[1 + artificial_column_index:1 + artificial_column_index + num_artificial_variables]

    # Checking for any common values between basis and artificial variables
    common_variables = [var for var in artificial_vars if var in basis]

    # If there exists common values, we check the row with corresponding index in basis.
    # If the RHS value is 0, the iterations continue, else the problem is infeasible and program terminated.
    if common_variables:
        # Iterating through the common values
        for var in common_variables:
            to_check = basis.index(var)
            if new_array[to_check][-1] != 0:
                result += "<br>The problem is infeasible"
                return result

    result += "<br><br>Since all values of Zj-Cj are <=0, optimal tableau has been found for Phase 1, we now proceed to phase 2"

    # Finding indices of all artificial variables to delete
    to_delete = [i for i in range(1 + artificial_column_index, 1 + artificial_column_index + num_artificial_variables)]

    # Removing artificial variables from tableau matrix
    new_array = np.delete(new_array, to_delete, axis=1)

    # Replacing cost row with original objective function values
    new_array[0, 1:1+len(c)] = -c

    # Removing artificial variable names from cost_variable array
    cost_variables = np.delete(cost_variables, to_delete)

    result += "<br><br>After removing artificial variables for phase 2<br>"
    result += print_tableau()

    # Preparing tableau for 2nd phase by making all the basis elements in basis array 0
    for ele_index, ele in enumerate(new_array[0][1:]):
        # If value in 0th cost_variable row of new_array is in basis, perform row operations to make it 0
        if(f"x{ele_index}" in basis):
            for row in new_array[1:]:
                if row[ele_index] != 0:
                    new_array[0][1:] -= new_array[0][ele_index] * row[1:]

    result += "<br><br>After preparing tableau for phase 2<br>"
    result += print_tableau()

    result += "<br><br>Phase 2"

    # Performing 2nd phase iterations
    result += tableau_iter()

    result += "<br>All Zj-Cj values are <=0, hence we have arrived at optimal solution"

    # Display optimal solution
    result += f"<br><br>Objective value: {new_array[0][-1]}"
    for val in cost_variables[1:1+num_variables]:
        if val in basis:
            index = basis.index(val)
            result += f"<br>{val} = {new_array[index][-1]}"
        else:
            result += f"<br>{val} = 0"

    return result

if __name__ == '__main__':
    app.run(debug=True)