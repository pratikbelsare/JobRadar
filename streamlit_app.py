"""Product-focused Streamlit client for the local JobRadar API."""

from __future__ import annotations

import html
import json
import os
from datetime import datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st

from job_intelligence.ui_presenters import (
    SORT_OPTIONS,
    VIEW_OPTIONS,
    JobPayload,
    company_name,
    dashboard_summary,
    eligibility_status,
    filter_jobs,
    humanize,
    primary_score,
    role_family,
    role_family_label,
    sort_jobs,
    table_rows,
)

API_URL = os.environ.get("JOB_INTELLIGENCE_API_URL", "http://127.0.0.1:8000").rstrip("/")


class ApiRequestError(RuntimeError):
    """A concise, user-safe API failure."""


def _request(
    path: str,
    method: str = "GET",
    payload: dict[str, object] | None = None,
) -> object:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(
        f"{API_URL}{path}",
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8")).get("detail")
        except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
            detail = None
        raise ApiRequestError(str(detail or f"Request failed with status {exc.code}")) from exc
    except (URLError, TimeoutError) as exc:
        raise ApiRequestError("The JobRadar API is not available right now.") from exc


@st.cache_data(ttl=15, show_spinner=False)
def _get(path: str) -> object:
    return _request(path)


def _refresh_data() -> None:
    _get.clear()


def _set_status(job_id: str, action: str, success_message: str) -> None:
    try:
        _request(f"/jobs/{job_id}/{action}", method="POST")
    except ApiRequestError as exc:
        st.warning(f"Status could not be updated: {exc}")
        return
    _refresh_data()
    st.session_state["flash_message"] = success_message
    st.rerun()


def _apply_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container { max-width: 1280px; padding-top: 2rem; padding-bottom: 4rem; }
        [data-testid="stSidebar"] { border-right: 1px solid rgba(128,128,128,.18); }
        [data-testid="stMetric"] {
            background: rgba(128,128,128,.055);
            border: 1px solid rgba(128,128,128,.14);
            border-radius: .75rem;
            padding: .85rem 1rem;
        }
        [data-testid="stMetricLabel"] { font-size: .78rem; letter-spacing: .02em; }
        .jr-eyebrow {
            color: #64748b; font-size: .78rem; font-weight: 650;
            letter-spacing: .08em; text-transform: uppercase;
        }
        .jr-subtitle { color: #64748b; margin-top: -.35rem; margin-bottom: 1.25rem; }
        .jr-badges { display: flex; flex-wrap: wrap; gap: .45rem; margin: .25rem 0 1rem; }
        .jr-badge {
            border-radius: 999px; padding: .23rem .65rem;
            font-size: .76rem; font-weight: 650; line-height: 1.35;
        }
        .jr-core, .jr-eligible, .jr-shortlisted { background: #dcfce7; color: #166534; }
        .jr-adjacent, .jr-stretch, .jr-maybe { background: #fef3c7; color: #92400e; }
        .jr-severe, .jr-dismissed { background: #fee2e2; color: #991b1b; }
        .jr-applied { background: #dbeafe; color: #1e40af; }
        .jr-new, .jr-unknown { background: #e2e8f0; color: #334155; }
        .jr-section-note { color: #64748b; font-size: .9rem; margin-top: -.5rem; }
        .jr-list { margin: .2rem 0 .8rem 0; padding-left: 1.2rem; }
        .jr-list li { margin-bottom: .35rem; }
        div[data-testid="stButton"] > button { border-radius: .55rem; }
        div[data-testid="stLinkButton"] > a { border-radius: .55rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _badge(label: str, style: str) -> str:
    return f'<span class="jr-badge jr-{style}">{html.escape(label)}</span>'


def _status_style(status: str) -> str:
    allowed = {"new", "shortlisted", "applied", "dismissed", "maybe"}
    return status if status in allowed else "unknown"


def _eligibility_style(status: str) -> str:
    return {
        "eligible": "eligible",
        "stretch": "stretch",
        "severe_stretch": "severe",
    }.get(status, "unknown")


def _family_style(family: str) -> str:
    return family if family in {"core", "adjacent"} else "unknown"


def _format_date(value: object) -> str:
    if not isinstance(value, str) or not value:
        return "Not available"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return parsed.strftime("%d %b %Y")


def _render_bullets(items: list[object], empty_message: str) -> None:
    cleaned = [str(item) for item in items if str(item).strip()]
    if not cleaned:
        st.caption(empty_message)
        return
    markup = "".join(f"<li>{html.escape(item)}</li>" for item in cleaned)
    st.markdown(f'<ul class="jr-list">{markup}</ul>', unsafe_allow_html=True)


def _render_dashboard(jobs: list[JobPayload]) -> None:
    summary = dashboard_summary(jobs)
    columns = st.columns(5)
    columns[0].metric("Matching jobs", summary["matching_jobs"])
    columns[1].metric("New", summary["new_jobs"])
    columns[2].metric("Shortlisted", summary["shortlisted"])
    columns[3].metric("Applied", summary["applied"])
    columns[4].metric("Companies", summary["companies"])


def _clear_filters() -> None:
    for key in (
        "job_search",
        "location_filter",
        "company_filter",
        "family_filter",
        "eligibility_filter",
        "status_filter",
        "job_sort",
    ):
        st.session_state.pop(key, None)


def _render_job_browser(jobs: list[JobPayload]) -> None:
    _render_dashboard(jobs)
    st.write("")
    view = st.radio(
        "View",
        VIEW_OPTIONS,
        horizontal=True,
        label_visibility="collapsed",
        key="job_view",
    )

    company_options = sorted({company_name(item) for item in jobs})
    family_options = sorted({role_family(item) for item in jobs})
    eligibility_options = sorted({eligibility_status(item) for item in jobs})
    status_options = sorted({str(item["job"].get("status", "new")) for item in jobs})

    st.sidebar.markdown("### Refine jobs")
    search_query = st.sidebar.text_input(
        "Search",
        placeholder="Title, company, or location",
        key="job_search",
    )
    selected_companies = st.sidebar.multiselect(
        "Company",
        company_options,
        format_func=str,
        key="company_filter",
    )
    selected_families = st.sidebar.multiselect(
        "Role family",
        family_options,
        format_func=role_family_label,
        key="family_filter",
    )
    selected_eligibility = st.sidebar.multiselect(
        "Eligibility",
        eligibility_options,
        format_func=humanize,
        key="eligibility_filter",
    )
    selected_statuses = st.sidebar.multiselect(
        "Workflow status",
        status_options,
        format_func=humanize,
        key="status_filter",
    )
    location_query = st.sidebar.text_input(
        "Location contains",
        placeholder="Remote, London, Bengaluru…",
        key="location_filter",
    )
    sort_by = st.sidebar.selectbox("Sort by", SORT_OPTIONS, key="job_sort")
    if st.sidebar.button("Clear filters", width="stretch"):
        _clear_filters()
        st.rerun()

    filtered = filter_jobs(
        jobs,
        view=view,
        companies=set(selected_companies),
        role_families=set(selected_families),
        eligibility=set(selected_eligibility),
        statuses=set(selected_statuses),
        location_query=location_query,
        search_query=search_query,
    )
    filtered = sort_jobs(filtered, sort_by)
    st.markdown("### Opportunities")
    st.caption(f"Showing {len(filtered)} of {len(jobs)} matching jobs · Select a row for details")
    if not filtered:
        st.info("No jobs match these filters. Clear a filter or choose another view.")
        return

    rows = table_rows(filtered)
    event = st.dataframe(
        rows,
        column_order=(
            "Job",
            "Company",
            "Relevance",
            "Eligibility",
            "Gap",
            "Family",
            "Status",
            "Location",
        ),
        hide_index=True,
        width="stretch",
        height=min(500, 40 + len(rows) * 35),
        on_select="rerun",
        selection_mode="single-row",
        key="job_results_table",
        column_config={
            "Job": st.column_config.TextColumn("Job", width="large"),
            "Company": st.column_config.TextColumn("Company", width="medium"),
            "Relevance": st.column_config.TextColumn("Relevance", width="small"),
            "Eligibility": st.column_config.TextColumn("Eligibility", width="medium"),
            "Gap": st.column_config.TextColumn("Experience gap", width="small"),
            "Family": st.column_config.TextColumn("Role family", width="small"),
            "Status": st.column_config.TextColumn("Status", width="small"),
            "Location": st.column_config.TextColumn("Location", width="large"),
        },
    )
    selected_rows = event.selection.rows
    if selected_rows:
        st.session_state["selected_job_id"] = rows[selected_rows[0]]["job_id"]
    visible_ids = {str(item["job"]["id"]) for item in filtered}
    selected_id = st.session_state.get("selected_job_id")
    if selected_id not in visible_ids:
        selected_id = str(filtered[0]["job"]["id"])
        st.session_state["selected_job_id"] = selected_id

    st.divider()
    try:
        detail = _get(f"/jobs/{selected_id}")
    except ApiRequestError as exc:
        st.warning(f"This job could not be loaded: {exc}")
        return
    if isinstance(detail, dict):
        _render_job_detail(detail)


def _render_job_detail(detail: JobPayload) -> None:
    job = detail.get("job") or {}
    company = detail.get("company") or {}
    match = detail.get("match") or {}
    explanation = detail.get("explanation")
    eligibility = match.get("eligibility") or {}
    status = str(job.get("status") or "new")
    family = str(match.get("role_family_class") or "unknown")
    eligibility_value = str(eligibility.get("status") or "unknown")
    relevance = primary_score(detail)
    job_id = str(job.get("id"))

    title_column, link_column = st.columns([4, 1])
    with title_column:
        st.markdown('<div class="jr-eyebrow">Selected opportunity</div>', unsafe_allow_html=True)
        st.header(str(job.get("title") or "Untitled role"))
        st.markdown(
            f"**{html.escape(str(company.get('name') or 'Unknown company'))}** · "
            f"{html.escape(str(job.get('location') or 'Location not specified'))}"
        )
        st.markdown(
            '<div class="jr-badges">'
            + _badge(role_family_label(family), _family_style(family))
            + _badge(humanize(eligibility_value), _eligibility_style(eligibility_value))
            + _badge(humanize(status), _status_style(status))
            + "</div>",
            unsafe_allow_html=True,
        )
    with link_column:
        st.write("")
        st.write("")
        if job.get("source_url"):
            st.link_button(
                "Open official posting",
                str(job["source_url"]),
                width="stretch",
                type="primary",
            )

    gap = eligibility.get("experience_gap_years")
    metrics = st.columns(4)
    metrics[0].metric("Career relevance", f"{relevance:.0%}")
    metrics[1].metric("Eligibility", humanize(eligibility_value))
    metrics[2].metric(
        "Experience gap",
        f"+{float(gap):.1f} years" if gap is not None and float(gap) > 0 else "No stated gap",
    )
    metrics[3].metric("Role family", role_family_label(family))
    if eligibility_value == "severe_stretch":
        st.warning(
            "Strong career alignment can still be difficult to act on immediately. "
            "This role has a significant experience or seniority stretch."
        )
    elif eligibility_value == "stretch":
        st.info("This role is relevant, with a manageable experience or seniority stretch.")
    elif eligibility_value == "eligible":
        st.success("The available evidence indicates that this role is currently feasible.")

    action_columns = st.columns([1, 1, 1, 2])
    if action_columns[0].button(
        "Shortlist",
        width="stretch",
        disabled=status == "shortlisted",
        type="primary" if status != "shortlisted" else "secondary",
        key=f"shortlist-{job_id}",
    ):
        _set_status(job_id, "shortlist", "Job added to your shortlist.")
    if action_columns[1].button(
        "Mark applied",
        width="stretch",
        disabled=status == "applied",
        key=f"applied-{job_id}",
    ):
        _set_status(job_id, "applied", "Job marked as applied.")
    if action_columns[2].button(
        "Dismiss",
        width="stretch",
        disabled=status == "dismissed",
        key=f"dismiss-{job_id}",
    ):
        _set_status(job_id, "dismiss", "Job dismissed from recommendations.")
    action_columns[3].caption(
        f"Current status: **{humanize(status)}**. Changes are saved through the JobRadar API."
    )

    st.write("")
    _render_explanation(job_id, explanation)
    st.write("")
    _render_evidence(match, eligibility)
    st.write("")
    _render_job_information(job, match)
    _render_feedback(job_id, detail.get("feedback"))


def _render_explanation(job_id: str, explanation: object) -> None:
    with st.container(border=True):
        st.subheader("Why JobRadar recommends this")
        st.markdown(
            (
                '<div class="jr-section-note">Grounded in the persisted candidate and job '
                "evidence.</div>"
            ),
            unsafe_allow_html=True,
        )
        if not isinstance(explanation, dict):
            st.info(
                "A grounded explanation is not available yet. The relevance and eligibility "
                "assessment above is still valid."
            )
            if st.button("Try generating explanation", key=f"explain-{job_id}"):
                try:
                    _request(f"/jobs/{job_id}/explanation", method="POST", payload={})
                except ApiRequestError:
                    st.warning(
                        "The explanation could not be generated right now. The job and its "
                        "match remain available."
                    )
                else:
                    _refresh_data()
                    st.rerun()
            return

        _render_bullets(
            list(explanation.get("why_fit") or []),
            "No concise fit summary is available.",
        )
        strongest, gaps = st.columns(2)
        with strongest:
            st.markdown("**Strongest matching evidence**")
            _render_bullets(
                list(explanation.get("strongest_matches") or []),
                "No strongest-match list was provided.",
            )
        with gaps:
            st.markdown("**Potential gaps**")
            _render_bullets(
                list(explanation.get("potential_gaps") or []),
                "No specific evidence gaps were identified.",
            )
        alignment = explanation.get("experience_alignment")
        if alignment:
            st.markdown(f"**Experience alignment:** {html.escape(str(alignment))}")
        considerations = list(explanation.get("important_considerations") or [])
        if considerations:
            st.markdown("**Important considerations**")
            _render_bullets(considerations, "")


def _render_evidence(match: dict[str, Any], eligibility: dict[str, Any]) -> None:
    components = (match.get("evidence") or {}).get("components") or {}
    skill_evidence = components.get("skill") or {}
    role_evidence = components.get("role") or {}
    location_evidence = components.get("location") or {}
    stage1 = (match.get("evidence") or {}).get("stage1") or {}
    stage1_rules = stage1.get("rules") or []
    location_rule = next(
        (rule for rule in stage1_rules if rule.get("rule") == "location"),
        None,
    )

    st.subheader("Fit and feasibility")
    strengths, gaps = st.columns(2)
    with strengths:
        with st.container(border=True):
            st.markdown("#### Matching evidence")
            matched_skills = [
                *list(skill_evidence.get("matched_required_skills") or []),
                *list(skill_evidence.get("matched_preferred_skills") or []),
            ]
            _render_bullets(
                matched_skills,
                "No exact skill-name overlap was identified; other semantic evidence may apply.",
            )
            matched_role = role_evidence.get("matched_role")
            if matched_role:
                st.caption(f"Role alignment: {matched_role}")
            responsibility = match.get("responsibility_match")
            if responsibility is not None:
                st.caption(f"Responsibility overlap: {float(responsibility):.0%}")
    with gaps:
        with st.container(border=True):
            st.markdown("#### Gaps and constraints")
            missing_skills = [
                *list(skill_evidence.get("required_skills_without_evidence") or []),
                *list(skill_evidence.get("preferred_skills_without_evidence") or []),
            ]
            _render_bullets(
                [f"{item} — no evidence found" for item in missing_skills],
                "No explicit skill-evidence gaps were identified.",
            )
            hard_blocks = list(eligibility.get("hard_block_reasons") or [])
            _render_bullets(hard_blocks, "No hard blockers were identified.")
            if location_rule and location_rule.get("reason"):
                st.caption(str(location_rule["reason"]))
            elif location_evidence.get("matched_location") is False:
                st.caption("The job location does not match a configured preferred location.")

    with st.expander("Advanced match details"):
        relevance_evidence = match.get("relevance_evidence") or {}
        relevance_components = relevance_evidence.get("components") or {}
        component_rows = []
        for name in ("role", "skill", "responsibility", "domain"):
            evidence = relevance_components.get(name) or {}
            component_rows.append(
                {
                    "Component": humanize(name),
                    "Score": evidence.get("raw_score"),
                    "Available": bool(evidence.get("available")),
                    "Weight": evidence.get("weight"),
                    "Contribution": evidence.get("weighted_contribution"),
                }
            )
        st.dataframe(component_rows, hide_index=True, width="stretch")
        legacy = match.get("legacy_final_score")
        if legacy is None:
            legacy = match.get("final_score")
        st.caption(
            f"Legacy blended score: {float(legacy):.3f} · "
            f"Ranking version: {match.get('ranking_version', 'unknown')}"
        )
        if stage1.get("reasons"):
            st.markdown("**Stage-1 decision trail**")
            _render_bullets(list(stage1["reasons"]), "")


def _render_job_information(job: dict[str, Any], match: dict[str, Any]) -> None:
    components = (match.get("evidence") or {}).get("components") or {}
    experience = components.get("experience") or {}
    minimum = experience.get("job_min_years")
    maximum = experience.get("job_max_years")
    if minimum is None and maximum is None:
        requirement = "Not clearly stated"
    elif maximum is None:
        requirement = f"{float(minimum):g}+ years"
    elif minimum is None:
        requirement = f"Up to {float(maximum):g} years"
    else:
        requirement = f"{float(minimum):g}–{float(maximum):g} years"

    st.subheader("Job information")
    columns = st.columns(4)
    columns[0].metric("Employment type", humanize(str(job.get("employment_type") or "unknown")))
    columns[1].metric("Experience requested", requirement)
    columns[2].metric("Seniority", humanize(str(job.get("seniority") or "unknown")))
    columns[3].metric("First discovered", _format_date(job.get("first_seen_at")))
    with st.expander("Read job description"):
        st.markdown(str(job.get("description") or "No description is available."))


def _render_feedback(job_id: str, feedback: object) -> None:
    with st.expander("Rate this recommendation"):
        existing = feedback if isinstance(feedback, list) else []
        if existing:
            latest = existing[-1]
            st.caption(f"Latest saved rating: {latest.get('label', 'Unknown')}")
        options = {
            "Irrelevant": 0,
            "Weak fit": 1,
            "Worth reviewing": 2,
            "Strong fit": 3,
        }
        label = st.selectbox(
            "How useful is this recommendation?",
            list(options),
            key=f"rate-{job_id}",
        )
        notes = st.text_input("Optional note", key=f"notes-{job_id}")
        if st.button("Save rating", key=f"feedback-{job_id}"):
            payload: dict[str, object] = {"label": options[label]}
            if notes.strip():
                payload["notes"] = notes.strip()
            try:
                _request(f"/jobs/{job_id}/feedback", method="POST", payload=payload)
            except ApiRequestError as exc:
                st.warning(f"Feedback could not be saved: {exc}")
            else:
                _refresh_data()
                st.success("Feedback saved.")


def _render_profile(profile: dict[str, Any]) -> None:
    st.markdown('<div class="jr-eyebrow">Candidate preferences</div>', unsafe_allow_html=True)
    st.title("Candidate profile")
    st.markdown(
        (
            '<div class="jr-subtitle">Manual profile values remain authoritative and are '
            "used by filtering and matching.</div>"
        ),
        unsafe_allow_html=True,
    )
    with st.form("candidate-profile"):
        roles = st.text_area(
            "Target roles",
            ", ".join(profile.get("target_roles") or []),
            help="Comma-separated role names.",
        )
        locations = st.text_area(
            "Preferred locations",
            ", ".join(profile.get("preferred_locations") or []),
            help="Comma-separated locations.",
        )
        experience = st.number_input(
            "Years of experience",
            min_value=0.0,
            value=float(profile.get("experience_years") or 0.0),
            step=0.1,
        )
        skills = st.text_area(
            "Skills",
            ", ".join(profile.get("skills") or []),
            help="Comma-separated skills; detailed evidence remains preserved.",
        )
        domains = st.text_area(
            "Domains",
            ", ".join(profile.get("domains") or []),
            help="Comma-separated areas of domain experience.",
        )
        work_modes = ["No preference", "Remote", "Hybrid", "Onsite", "Flexible", "Unknown"]
        current_mode = profile.get("work_mode")
        current_label = humanize(str(current_mode)) if current_mode else "No preference"
        mode_index = work_modes.index(current_label) if current_label in work_modes else 0
        work_mode = st.selectbox("Preferred work mode", work_modes, index=mode_index)
        submitted = st.form_submit_button("Save profile", type="primary")
    if submitted:
        updated = dict(profile)
        updated["target_roles"] = _comma_values(roles)
        updated["preferred_locations"] = _comma_values(locations)
        updated["experience_years"] = experience
        updated["skills"] = _comma_values(skills)
        updated["domains"] = _comma_values(domains)
        updated["work_mode"] = None if work_mode == "No preference" else work_mode.casefold()
        try:
            _request("/profile", method="PUT", payload=updated)
        except ApiRequestError as exc:
            st.warning(f"Profile could not be saved: {exc}")
        else:
            _refresh_data()
            st.success("Candidate profile saved.")


def _comma_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    st.set_page_config(
        page_title="JobRadar",
        page_icon="JR",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _apply_styles()
    st.sidebar.markdown("## JobRadar")
    st.sidebar.caption("Personal job intelligence")
    page = st.sidebar.radio(
        "Workspace",
        ("Jobs", "Candidate profile"),
        label_visibility="collapsed",
    )
    st.sidebar.divider()

    flash = st.session_state.pop("flash_message", None)
    if flash:
        st.success(str(flash))

    if page == "Candidate profile":
        try:
            profile = _get("/profile")
        except ApiRequestError as exc:
            st.error(str(exc))
            st.code("python -m uvicorn job_intelligence.api:create_app --factory --reload")
            return
        if isinstance(profile, dict):
            _render_profile(profile)
        return

    st.markdown('<div class="jr-eyebrow">Your opportunity radar</div>', unsafe_allow_html=True)
    st.title("Find the roles worth your attention")
    st.markdown(
        (
            '<div class="jr-subtitle">Career relevance shows direction. Eligibility shows '
            "what is realistic right now.</div>"
        ),
        unsafe_allow_html=True,
    )
    try:
        with st.spinner("Loading recommendations…"):
            payload = _get("/jobs?min_match_score=0&sort_by=score")
    except ApiRequestError as exc:
        st.error(str(exc))
        st.caption("Start the local API, then refresh this page.")
        st.code("python -m uvicorn job_intelligence.api:create_app --factory --reload")
        return
    if not isinstance(payload, list):
        st.warning("JobRadar returned an unexpected response. Please refresh the page.")
        return
    jobs = [item for item in payload if isinstance(item, dict)]
    if not jobs:
        st.info("No matched jobs are available yet. Run a local scan and refresh this page.")
        return
    _render_job_browser(jobs)


if __name__ == "__main__":
    main()
