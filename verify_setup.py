"""Smoke test: verify all dependencies are installed correctly."""
import sys

def check(name, import_name=None):
    try:
        mod = __import__(import_name or name)
        ver = getattr(mod, "__version__", "ok")
        print(f"  ✓ {name:20s} {ver}")
        return True
    except ImportError:
        print(f"  ✗ {name:20s} NOT FOUND")
        return False

print(f"\nPython {sys.version}\n")
print("Package check:")
all_ok = all([
    check("pandas"),
    check("numpy"),
    check("yfinance"),
    check("scipy"),
    check("cvxpy"),
    check("matplotlib"),
    check("seaborn"),
    check("jupyter"),
    check("pytest"),
])

# Test CVXPY solver
print("\nCVXPY solver check:")
import cvxpy
solvers = cvxpy.installed_solvers()
print(f"  Available solvers: {solvers}")
has_solver = any(s in solvers for s in ["CLARABEL","SCS","OSQP","ECOS"])
print(f"  Usable solver found: {'✓ Yes' if has_solver else '✗ No'}")

# Test yfinance quick download
print("\nyfinance connection test:")
import yfinance as yf
ticker = yf.Ticker("CSPX.L")
hist = ticker.history(period="5d")
print(f"  Downloaded {len(hist)} rows for CSPX.L")
print(f"  Latest close: {hist['Close'].iloc[-1]:.2f}")

print("\n" + ("═" * 45))
print("ALL CHECKS PASSED ✓" if all_ok and has_solver else "SOME CHECKS FAILED ✗")
print("═" * 45)