import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from scipy.optimize import linprog

@pytest.fixture(scope="module")
def driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    s = Service()
    driver_instance = webdriver.Chrome(service=s, options=options)
    yield driver_instance
    driver_instance.quit()

def run_test_case(driver, A, equalities, b, c):
    driver.get("http://localhost:5000")  # Replace with your website's URL

    # Input values
    driver.find_element(By.NAME, "A").send_keys(A)
    driver.find_element(By.NAME, "equalities").send_keys(equalities)
    driver.find_element(By.NAME, "b").send_keys(b)
    driver.find_element(By.NAME, "c").send_keys(c)

    # Submit the form
    driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()

    # Wait for the result to be displayed
    result = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, ".result"))
    )

    # Solve the linear programming problem using SciPy
    c = [float(x) for x in c.split(',')]
    A_ub = [[float(x) for x in row.split(',')] for row in A.split('\n') if ',' in row]
    b_ub = [float(x) for x in b.split(',') if 'less than' in equalities] 
    A_eq = [[float(x) for x in row.split(',')] for row in A.split('\n') if ',' in row and 'equal' in equalities]
    b_eq = [float(x) for x in b.split(',') if 'equal' in equalities]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method='simplex')
    print(res)

    # Define a tolerance for comparison
    tolerance = 1e-6

    print(result.text.split(": "))
    # Assert the expected outcome
    assert abs(float(result.text.split(": ")[1]) - res.fun) < tolerance
    for i in range(len(res.x)):
        assert abs(float(result.text.split("\n")[i+1].split(" = ")[1]) - res.x[i]) < tolerance

    print(f"Test case with inputs:\nA={A}\nequalities={equalities}\nb={b}\nc={c}\npassed!")

# Example usage
def test_case_1(driver):
    A = "1, 1, 1\n1, 2, 3\n3, 2, 1"
    equalities = "equal,less than,less than"
    b = "100, 240, 270"
    c = "3, 4, 2"
    run_test_case(driver, A, equalities, b, c)

def test_case_2(driver):
    A = "1, 2, 3\n4, 5, 6"
    equalities = "less than,equal"
    b = "10, 20"
    c = "1, 2, 3"
    expected_optimal_value = 10.0
    expected_x = [0.0, 0.0, 3.33333]
    run_test_case(driver, A, equalities, b, c)
    
