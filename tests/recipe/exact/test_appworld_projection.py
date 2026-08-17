from recipe.exact.appworld_adapter import compile_json_api_action


def test_json_api_projection_compiles_one_fixed_shape_call():
    action = compile_json_api_action('<think>inspect docs</think><action>{"app":"api_docs","api":"show_api_doc","arguments":{"app_name":"spotify","include_private":false}}</action>')
    assert action == "print(apis.api_docs.show_api_doc(**{'app_name': 'spotify', 'include_private': False}))"


def test_json_api_projection_rejects_code_and_multiple_calls():
    invalid_actions = [
        '<think>x</think><action>{"app":"api_docs","api":"show_app_descriptions","arguments":{}}</action><action>{"app":"supervisor","api":"complete_task","arguments":{}}</action>',
        "<think>x</think><code>print(apis.supervisor.complete_task())</code>",
        '<think>x</think><action>{"app":"__class__","api":"mro","arguments":{}}</action>',
        '<think>x</think><action>{"api":"show_profile","app":"supervisor","arguments":{}}</action>',
        '<think>x</think><action>{"app":"supervisor","api":"show_profile","api":"complete_task","arguments":{}}</action>',
        '<think>x</think><action>{"app":"supervisor","api":"show_profile","arguments":{}}</action>junk',
    ]
    for invalid_action in invalid_actions:
        try:
            compile_json_api_action(invalid_action)
        except ValueError:
            pass
        else:
            raise AssertionError("structured adapter accepted an unsafe action")
