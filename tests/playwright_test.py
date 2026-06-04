from playwright.sync_api import sync_playwright
import requests

BASE = "http://127.0.0.1:8080"

def check_health():
    r = requests.get(f"{BASE}/health", timeout=10)
    print('health', r.status_code, r.json())
    return r.status_code == 200


def run_ui_checks():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        print('Opening login...')
        resp = page.goto(f"{BASE}/login", timeout=15000)
        assert resp and resp.ok, f"Login page not reachable ({resp})"

        print('Submitting login form...')
        page.fill('input[name="username"]', 'admin')
        page.fill('input[name="password"]', 'admin')
        page.click('button[type="submit"]')
        page.wait_for_load_state('networkidle')
        # After login we expect to be redirected to dashboard (/)
        assert page.url.startswith(BASE + '/'), f"Unexpected URL after login: {page.url}"
        print('Login OK, on', page.url)

        # Open settings page
        resp = page.goto(f"{BASE}/settings")
        assert resp and resp.ok, 'Settings page not reachable'
        print('Settings page OK')

        # Open dashboard
        resp = page.goto(f"{BASE}/")
        assert resp and resp.ok, 'Dashboard not reachable'
        content = page.content()
        assert '策略' in content or 'strategy' in content.lower(), 'Dashboard content looks unexpected'
        print('Dashboard content check passed')

        browser.close()
        return True


if __name__ == '__main__':
    ok = True
    try:
        ok = ok and check_health()
    except Exception as e:
        print('Health check failed', e)
        ok = False
    try:
        ok = ok and run_ui_checks()
    except AssertionError as e:
        print('UI check assertion failed:', e)
        ok = False
    except Exception as e:
        print('UI check error:', e)
        ok = False

    print('ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED')
    if not ok:
        raise SystemExit(2)
