"""Small personal Streamlit client for the local JobRadar API."""

from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st

API_URL = os.environ.get("JOB_INTELLIGENCE_API_URL", "http://127.0.0.1:8000").rstrip("/")


def _request(path: str, method: str = "GET", payload: dict[str, object] | None = None) -> object:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(
        f"{API_URL}{path}",
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as exc:
        st.error(f"API request failed: {exc}")
        return None


st.set_page_config(page_title="JobRadar", layout="wide")
st.title("JobRadar")
st.caption(f"API: {API_URL}")

jobs = _request("/jobs?sort_by=score")
profile = _request("/profile")

jobs_tab, profile_tab = st.tabs(["Jobs", "Candidate profile"])
with jobs_tab:
    if isinstance(jobs, list):
        choices = {
            f"{item['job']['title']} — "
            f"{item['company']['name'] if item.get('company') else 'Unknown company'}": item
            for item in jobs
        }
        if not choices:
            st.info("No persisted jobs yet.")
        else:
            selected_label = st.selectbox("Jobs", list(choices))
            selected = choices[selected_label]
            job_id = selected["job"]["id"]
            detail = _request(f"/jobs/{job_id}")
            if isinstance(detail, dict):
                job = detail["job"]
                match = detail.get("match")
                st.subheader(job["title"])
                st.write(f"{job.get('location') or 'Location not specified'} · {job['status']}")
                if match:
                    st.metric("Match score", f"{match['final_score']:.0%}")
                st.markdown(job["description"])
                st.link_button("Open career-site listing", job["source_url"])
                columns = st.columns(3)
                if columns[0].button("Shortlist"):
                    _request(f"/jobs/{job_id}/shortlist", method="POST")
                    st.rerun()
                if columns[1].button("Dismiss"):
                    _request(f"/jobs/{job_id}/dismiss", method="POST")
                    st.rerun()
                if columns[2].button("Mark applied"):
                    _request(f"/jobs/{job_id}/applied", method="POST")
                    st.rerun()
                if st.button("Request explanation"):
                    explanation = _request(f"/jobs/{job_id}/explanation", method="POST", payload={})
                    if isinstance(explanation, dict):
                        st.json(explanation)
                feedback_options = {
                    "Irrelevant": 0,
                    "Weak fit": 1,
                    "Worth reviewing": 2,
                    "Strong fit": 3,
                }
                feedback_label = st.selectbox(
                    "Relevance feedback",
                    options=list(feedback_options),
                )
                if st.button("Submit feedback"):
                    _request(
                        f"/jobs/{job_id}/feedback",
                        method="POST",
                        payload={"label": feedback_options[feedback_label]},
                    )
                    st.success("Feedback saved")

with profile_tab:
    if isinstance(profile, dict):
        with st.form("profile"):
            roles = st.text_input("Target roles", ", ".join(profile.get("target_roles", [])))
            locations = st.text_input(
                "Preferred locations", ", ".join(profile.get("preferred_locations", []))
            )
            experience = st.number_input(
                "Years of experience",
                min_value=0.0,
                value=float(profile.get("experience_years", 0)),
            )
            submitted = st.form_submit_button("Save profile")
        if submitted:
            profile["target_roles"] = [item.strip() for item in roles.split(",") if item.strip()]
            profile["preferred_locations"] = [
                item.strip() for item in locations.split(",") if item.strip()
            ]
            profile["experience_years"] = experience
            updated = _request("/profile", method="PUT", payload=profile)
            if updated is not None:
                st.success("Profile saved")
