"""End-to-end tests of the dashboard (vidHMR_viewer.app) in headless Chromium.

PYTHONPATH=src .venv/bin/python -m pytest tests/test_dashboard_e2e.py -x -q
VIDHMR_E2E_GL=software ...   # WebGL in software (default: the GPU through Vulkan)
VIDHMR_E2E_ALL=1 ...         # also open the recording of every sequence (about 15 min)
"""

import json
import os
import socket
import threading
import time

import pytest
import uvicorn
from playwright.sync_api import sync_playwright

from vidHMR import paths
from vidHMR.datasets import pw3d
from vidHMR_viewer.app import build_app, worst_frame
from vidHMR_viewer.recording import recording_path

GL_ARGS = {
    "gpu": [
        "--use-angle=vulkan",
        "--enable-features=Vulkan",
        "--enable-gpu",
        "--ignore-gpu-blocklist",
    ],
    "software": ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
}[os.environ.get("VIDHMR_E2E_GL", "gpu")]
ROWS = pw3d.index()
STATE_JS = (
    "window.__dash.viewer.getState()"
    ".then(s => JSON.stringify(s, (k, v) => typeof v === 'bigint' ? Number(v) : v))"
)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def server():
    port = free_port()
    srv = uvicorn.Server(
        uvicorn.Config(build_app(), host="127.0.0.1", port=port, log_level="error")
    )
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    while not srv.started:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    thread.join(5)


@pytest.fixture(scope="session")
def browser():
    with sync_playwright() as p:
        b = p.chromium.launch(args=GL_ARGS)
        yield b
        b.close()


@pytest.fixture
def page(browser, server):
    pg = browser.new_page(viewport={"width": 1600, "height": 1000})
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(server + "/")
    pg.wait_for_function("window.__dash?.ready", timeout=30000)
    yield pg
    pg.close()
    assert not errors, errors


# -------------------------------------------------------------------- helpers
def viewer_state(pg) -> dict:
    return json.loads(pg.evaluate(STATE_JS))


def recording(pg) -> dict:
    recs = viewer_state(pg)["recordings"]
    assert len(recs) == 1, [r["storeId"] for r in recs]  # the previous one was closed
    return recs[0]


def wait_loaded(pg, name: str, timeout: float = 120) -> dict:
    """Wait until the viewer has the recording of `name` completely, and return it."""
    pg.wait_for_function("document.getElementById('status').hidden", timeout=timeout * 1000)
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = viewer_state(pg)
        recs = [r for r in st["recordings"] if r["storeId"].endswith(":" + name)]
        if recs and not st.get("loading"):
            return recs[0]
        pg.wait_for_timeout(250)
    raise TimeoutError(f"{name} did not load in {timeout} s")


def open_row(pg, table: str, name: str) -> dict:
    pg.locator(f'{table} tr[data-name="{name}"]').click()
    return wait_loaded(pg, name)


def table_names(pg, table: str = "#seq-table") -> list[str]:
    return pg.eval_on_selector_all(f"{table} tbody tr", "rs => rs.map(r => r.dataset.name)")


def current_frame(rec: dict) -> int:
    return rec["currentTime"]["time"]


def view_reports(pg) -> list:
    return [(v["name"], r["summary"]) for v in viewer_state(pg)["views"] for r in v["reports"]]


# -------------------------------------------------------------------- tables
def test_page_loads(page):
    tabs = page.eval_on_selector_all('[role="tab"]', "ts => ts.map(t => t.textContent)")
    assert tabs == ["Sequences", "SAM 3D Body", "Dataset summary"]
    assert len(table_names(page)) == len(ROWS) == 60
    assert page.locator("#empty").is_visible()
    chips = page.eval_on_selector_all("#subsets button", "bs => bs.map(b => b.textContent)")
    assert chips[0] == "All 60" and "Test 24" in chips


def test_filters(page):
    page.locator('#subsets button[data-key="pc"]').click()
    assert sorted(table_names(page)) == sorted(pw3d.PC_RANGES)

    page.locator('#subsets button[data-key="all"]').click()
    page.fill("#search", "courtyard_dancing")
    assert sorted(table_names(page)) == ["courtyard_dancing_00", "courtyard_dancing_01"]

    page.fill("#search", "")
    page.select_option("#actors", "2")
    two = [r["name"] for r in ROWS if r["actors"] == 2]
    assert sorted(table_names(page)) == sorted(two)

    page.fill("#search", "no such sequence")
    assert table_names(page) == []
    assert page.locator("#subset-note").text_content() == "No sequence matches the filters."


def test_sort(page):
    page.locator('#seq-table th[data-key="frames"]').click()
    frames = [r["frames"] for r in sorted(ROWS, key=lambda r: r["frames"])]
    shown = page.eval_on_selector_all(
        "#seq-table tbody tr td:nth-child(3)", "c => c.map(x => +x.textContent)"
    )
    assert shown == frames
    page.locator('#seq-table th[data-key="frames"]').click()
    shown = page.eval_on_selector_all(
        "#seq-table tbody tr td:nth-child(3)", "c => c.map(x => +x.textContent)"
    )
    assert shown == frames[::-1]


def test_summary_tab(page):
    page.locator('[role="tab"][data-tab="summary"]').click()
    assert page.locator("#tab-sequences").is_hidden()
    total = page.locator("#summary-table tbody tr").last.text_content()
    assert total.startswith("Total60")


# -------------------------------------------------------------------- the viewer
def test_open_and_play(page):
    name = "courtyard_arguing_00"
    rec = open_row(page, "#seq-table", name)
    seq = next(r for r in ROWS if r["name"] == name)
    timeline = rec["timelines"][0]
    assert timeline["name"] == "frame"
    assert timeline["timeRange"] == {"start": 0, "end": seq["frames"] - 1}
    assert page.locator("#meta").text_content().startswith(name)
    assert page.locator("#empty").is_hidden()
    assert not view_reports(page)

    views = {v["name"]: v["class"] for v in viewer_state(page)["views"]}
    assert views == {
        "Camera": "2D",
        "3D": "3D",
        "SAM 3D Body error (mm)": "TimeSeries",
        "SAM 3D Body root error (mm)": "TimeSeries",
    }

    # Real-time playback: about 30 frames per second.
    page.evaluate(
        "window.__dash.viewer.setTime({timeline: 'frame', sequence: 0, playState: 'playing'})"
    )
    t0 = time.time()
    page.wait_for_timeout(4000)
    rec = recording(page)
    rate = current_frame(rec) / (time.time() - t0)
    assert rec["playState"] == "playing"
    assert 24 <= rate <= 33, rate

    page.evaluate("window.__dash.viewer.setTime({timeline: 'frame', playState: 'paused'})")
    f = current_frame(recording(page))
    page.wait_for_timeout(1000)
    assert current_frame(recording(page)) == f


def test_status_and_diagnostics(page):
    name = "downtown_walkUphill_00"
    open_row(page, "#seq-table", name)  # waits for the hidden status and the full load
    assert page.locator("#status").is_hidden()
    page.wait_for_timeout(2500)  # one heartbeat
    log = paths.GENERATED / "logs" / "client.log"
    lines = [
        json.loads(x[x.index("{") :]) for x in log.read_text().splitlines()[-200:] if "{" in x
    ]
    kinds = {d["kind"] for d in lines if d.get("open", name) == name}
    assert {"start", "beat", "opened", "loaded"} <= kinds, kinds


def test_play_button(page):
    open_row(page, "#seq-table", "courtyard_arguing_00")
    assert page.locator("#controls").is_visible()
    assert page.locator("#playlabel").text_content() == "Play"

    page.locator("#play").click()
    page.wait_for_timeout(2000)
    rec = recording(page)
    assert rec["playState"] == "playing" and current_frame(rec) > 30
    assert page.locator("#playlabel").text_content() == "Pause"
    assert page.locator("#frame").text_content().startswith("Frame ")

    page.locator("#play").click()
    page.wait_for_timeout(500)
    f = current_frame(recording(page))
    page.wait_for_timeout(1000)
    assert recording(page)["playState"] == "paused" and current_frame(recording(page)) == f
    assert page.locator("#playlabel").text_content() == "Play"

    page.keyboard.press("Space")  # the page has the focus, not the viewer
    page.wait_for_timeout(1000)
    assert recording(page)["playState"] == "playing"

    page.locator("#restart").click()
    page.wait_for_timeout(300)
    assert current_frame(recording(page)) < 30

    # Scrub bar: a click at 3/4 of its width goes to about 3/4 of the sequence, paused.
    box = page.locator("#scrub").bounding_box()
    page.mouse.click(box["x"] + 0.75 * box["width"], box["y"] + box["height"] / 2)
    page.wait_for_timeout(500)
    rec = recording(page)
    assert rec["playState"] == "paused"
    assert abs(current_frame(rec) - 0.75 * 764) < 40, current_frame(rec)


def test_controls_at_small_height(browser, server):
    """At a small window height Rerun hides its own timeline: the page controls stay."""
    pg = browser.new_page(viewport={"width": 1638, "height": 765}, device_scale_factor=1.25)
    pg.goto(server + "/#seq=courtyard_arguing_00")
    pg.wait_for_function("window.__dash?.ready && window.__dash.viewer", timeout=60000)
    wait_loaded(pg, "courtyard_arguing_00")
    for el in ("#play", "#scrub", "#frame"):
        box = pg.locator(el).bounding_box()
        assert box and box["y"] + box["height"] <= 765, el
    pg.close()


def test_switch_sequences(page):
    for name in ("courtyard_arguing_00", "downtown_walkUphill_00", "courtyard_arguing_00"):
        open_row(page, "#seq-table", name)
        rec = recording(page)  # asserts that only one recording is open
        assert rec["storeId"].endswith(":" + name)
        assert page.locator("#seq-table tr.open").get_attribute("data-name") == name
    assert page.evaluate("location.hash") == "#seq=courtyard_arguing_00"


def test_pc_sequence_opens_at_range(page):
    name = "courtyard_dancing_01"
    open_row(page, "#seq-table", name)
    assert current_frame(recording(page)) == pw3d.PC_RANGES[name][0]


def test_sam3d_tab_opens_worst_frame(page):
    page.locator('[role="tab"][data-tab="sam3d"]').click()
    assert page.locator("#sam3d-totals tbody tr").count() >= 4
    assert "3dpw_train" in page.locator("#sam3d-caution").text_content()
    names = table_names(page, "#sam3d-table")
    assert len(names) == 60
    name = "downtown_walkUphill_00"
    open_row(page, "#sam3d-table", name)
    assert current_frame(recording(page)) == worst_frame(name)


def test_deep_link(browser, server):
    pg = browser.new_page(viewport={"width": 1600, "height": 1000})
    pg.goto(server + "/#seq=courtyard_arguing_00&frame=300")
    pg.wait_for_function("window.__dash?.ready && window.__dash.viewer", timeout=60000)
    wait_loaded(pg, "courtyard_arguing_00")
    assert current_frame(recording(pg)) == 300
    pg.close()


def test_phone_width(browser, server):
    pg = browser.new_page(viewport={"width": 390, "height": 844})
    pg.goto(server + "/")
    pg.wait_for_function("window.__dash?.ready", timeout=30000)
    assert pg.evaluate("document.documentElement.scrollWidth") <= 390
    pg.close()


@pytest.mark.skipif(not os.environ.get("VIDHMR_E2E_ALL"), reason="set VIDHMR_E2E_ALL=1")
@pytest.mark.parametrize("name", [r["name"] for r in ROWS])
def test_every_recording(page, name):
    assert recording_path(name).exists()
    rec = open_row(page, "#seq-table", name)
    seq = next(r for r in ROWS if r["name"] == name)
    # The full range, also after the load: a viewer that drops old data at its memory limit
    # would move the start of the range.
    assert rec["timelines"][0]["timeRange"] == {"start": 0, "end": seq["frames"] - 1}
    for f in (0, seq["frames"] // 2, seq["frames"] - 1):
        page.evaluate(f"window.__dash.viewer.setTime({{timeline: 'frame', sequence: {f}}})")
        page.wait_for_timeout(300)
        assert not view_reports(page), (f, view_reports(page))
    assert recording(page)["timelines"][0]["timeRange"]["start"] == 0
