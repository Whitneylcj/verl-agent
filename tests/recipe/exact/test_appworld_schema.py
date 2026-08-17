from recipe.exact.appworld_schema import (
    POLICY_CONTEXT_RESOURCE,
    appworld_factor_id,
    appworld_model_resource,
    build_appworld_effect_registry,
    compile_appworld_factor_reads,
    resolve_appworld_effect_schema,
)


class CharacterTokenizer:
    def decode(self, token_ids, **kwargs):
        assert kwargs["skip_special_tokens"] is True
        return "".join(chr(token_id) for token_id in token_ids)


EVALUATION_CODE = """
def evaluate(test, public_data, private_data, main_user, models, ground_truth_answer):
    active_tasks = models.end.supervisor.Task.all()
    predicted_answer = active_tasks[0].answer
    with test('''
        answers match
        '''):
        test.answer(predicted_answer, ground_truth_answer)
    with test("only allowed models changed"):
        test.case(models.changed_model_names(), "==", {"venmo.PaymentRequest"})
    added, _, _ = models.changed_records("venmo.PaymentRequest")
    with test("payment requests match"):
        test.case(added, "==", private_data.expected)
    with test("playlist remains"):
        test.case(models.end.spotify.Playlist.all(), "==", public_data.playlists)
    model_suffix = private_data.model_suffix
    with test("dynamic model lookup"):
        models.changed_records(f"phone.{model_suffix}")
"""


def test_compile_appworld_factor_reads_tracks_dataflow_and_fails_closed():
    resources = (
        appworld_model_resource("supervisor", "Task"),
        appworld_model_resource("venmo", "PaymentRequest"),
        appworld_model_resource("spotify", "Playlist"),
        appworld_model_resource("phone", "GlobalTextMessage"),
    )
    compiled = compile_appworld_factor_reads(EVALUATION_CODE, resources)

    assert compiled.read_sets[appworld_factor_id("answers match")] == (appworld_model_resource("supervisor", "Task"),)
    assert compiled.read_sets[appworld_factor_id("payment requests match")] == (appworld_model_resource("venmo", "PaymentRequest"),)
    assert compiled.read_sets[appworld_factor_id("playlist remains")] == (appworld_model_resource("spotify", "Playlist"),)
    opaque_ids = set(compiled.opaque_factor_ids)
    assert appworld_factor_id("only allowed models changed") not in opaque_ids
    assert appworld_factor_id("dynamic model lookup") in opaque_ids
    assert compiled.read_sets[appworld_factor_id("only allowed models changed")] == tuple(sorted(resources))
    assert compiled.read_sets[appworld_factor_id("dynamic model lookup")] == tuple(sorted(resources))

    task_specific = compile_appworld_factor_reads(
        EVALUATION_CODE,
        resources,
        known_values={"private_data": {"model_suffix": "GlobalTextMessage"}},
    )
    assert appworld_factor_id("dynamic model lookup") not in set(task_specific.opaque_factor_ids)
    assert task_specific.read_sets[appworld_factor_id("dynamic model lookup")] == (appworld_model_resource("phone", "GlobalTextMessage"),)


AUDITED_APPWORLD_REVISION = "a072b7a86e7c1d5b1d7175659d750ebb9b79f10a"


def _registry(version="0.2.0.dev0", revision=AUDITED_APPWORLD_REVISION):
    return build_appworld_effect_registry(
        api_docs={
            "venmo": {"create_payment_request": {}},
            "spotify": {"create_playlist": {}},
        },
        app_to_model_names={
            "admin": ("PaymentCard",),
            "file_system": ("File",),
            "gmail": ("Email",),
            "phone": ("Message",),
            "spotify": ("Playlist",),
            "venmo": ("Notification", "PaymentRequest"),
        },
        appworld_version=version,
        appworld_source_revision=revision,
    )


def test_effect_registry_uses_audited_service_closure_and_version_fallback():
    registry = _registry()
    assert registry["source_supported"] is True
    venmo_writes = set(registry["api_possible_write_sets"]["venmo.create_payment_request"])
    assert appworld_model_resource("venmo", "PaymentRequest") in venmo_writes
    assert appworld_model_resource("venmo", "Notification") in venmo_writes
    assert appworld_model_resource("gmail", "Email") in venmo_writes
    assert appworld_model_resource("spotify", "Playlist") not in venmo_writes

    fallback = _registry("9.9.9")
    assert fallback["version_supported"] is False
    assert set(fallback["api_possible_write_sets"]["venmo.create_payment_request"]) == set(fallback["all_model_resources"])
    wrong_source = _registry(revision="wrong")
    assert wrong_source["source_supported"] is False
    assert set(wrong_source["api_possible_write_sets"]["venmo.create_payment_request"]) == set(wrong_source["all_model_resources"])


def test_resolve_effect_schema_splits_selector_from_prefix_known_arguments():
    text = '<think>inspect</think><action>{"app":"venmo","api":"create_payment_request","arguments":{"user_email":"a@example.com","amount":3}}</action>'
    schema = resolve_appworld_effect_schema(_registry(), text, [ord(character) for character in text], CharacterTokenizer())

    assert schema["resolution_fallback"] is False
    assert [span["span_id"] for span in schema["spans"]] == [
        "appworld.selector",
        "appworld.arguments",
    ]
    selector, arguments = schema["spans"]
    assert selector["token_start"] == 0
    assert selector["token_end"] == arguments["token_start"]
    assert arguments["token_end"] == len(text)
    assert POLICY_CONTEXT_RESOURCE in selector["possible_write_set"]
    assert appworld_model_resource("spotify", "Playlist") in selector["possible_write_set"]
    assert appworld_model_resource("spotify", "Playlist") not in arguments["possible_write_set"]
    assert appworld_model_resource("venmo", "PaymentRequest") in arguments["possible_write_set"]


def test_invalid_action_resolves_to_conservative_whole_response_selector():
    text = "not valid"
    schema = resolve_appworld_effect_schema(_registry(), text, [ord(character) for character in text], CharacterTokenizer())
    assert schema["resolution_fallback"] is True
    assert len(schema["spans"]) == 1
    assert schema["spans"][0]["token_end"] == len(text)
