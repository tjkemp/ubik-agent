from ubikagent import Project
from ubikagent.introspection import get_methods


class DummyAgent:
    """Test class needed by `InstantiableProject` and `TestProject`."""
    pass


class NonInstantiableProject(Project):
    """Test class needed by `TestProject`."""
    pass


class InstantiableProject(Project):
    """Test class needed by `TestProject` and `TestIntrospection`."""

    ENV_ID = 'test-v0'
    AGENT_CLASS = DummyAgent

    def no_args(self):
        pass

    def pos_arg(self, argument):
        pass

    def pos_arg_with_explicit_type(self, argument: int):
        pass

    def kwarg_with_implicit_int_type(self, argument=1):
        pass

    def kwarg_with_default_none(self, argument=None):
        pass

    def kwarg_with_explicit_int_type(self, argument: int = 1):
        pass

    def kwarg_with_implicit_bool_type(self, argument=True):
        pass

    def kwarg_with_implicit_string_type(self, argument='a_string'):
        pass


class TestIntrospection:
    """Tests reading methods and arguments from `Project` and its subclasses
    to be used to generate command line help."""

    def setup_class(cls):
        cls.instance = InstantiableProject()
        cls.methods = get_methods(cls.instance)

    def test_project_method_without_args(self):

        method_name = 'no_args'
        argument = self.methods[method_name]
        assert argument == []

    def test_project_method_with_an_arg(self):

        method_name = 'pos_arg'
        expected_name = 'argument'
        expected_kwarg = False
        expected_default = None
        expected_type = None
        expected_doc = None

        arguments = self.methods[method_name]
        first_argument = arguments[0]
        assert first_argument == (expected_name, expected_kwarg, expected_default, expected_type, expected_doc)

    def test_project_method_with_an_arg_with_explicit_type(self):

        method_name = 'pos_arg_with_explicit_type'
        expected_name = 'argument'
        expected_default = None
        expected_type = int

        arguments = self.methods[method_name]
        first_argument = arguments[0]
        argument_name, is_kwarg, argument_default, argument_type, _ = first_argument
        assert argument_name == expected_name
        assert is_kwarg is False
        assert argument_default == expected_default
        assert argument_type == expected_type

    def test_project_method_default_none(self):

        method_name = 'kwarg_with_default_none'
        expected_name = 'argument'
        expected_default = None
        expected_type = None

        arguments = self.methods[method_name]
        first_argument = arguments[0]
        argument_name, is_kwarg, argument_default, argument_type, _ = first_argument
        assert argument_name == expected_name
        assert is_kwarg is True
        assert argument_default == expected_default
        assert argument_type == expected_type

    def test_project_method_with_int_default(self):

        method_name = 'kwarg_with_implicit_int_type'
        expected_name = 'argument'
        expected_default = 1
        expected_type = int

        arguments = self.methods[method_name]
        first_argument = arguments[0]
        argument_name, is_kwarg, argument_default, argument_type, _ = first_argument
        assert argument_name == expected_name
        assert is_kwarg is True
        assert argument_default == expected_default
        assert argument_type == expected_type

    def test_project_method_with_int_type(self):

        method_name = 'kwarg_with_explicit_int_type'
        expected_name = 'argument'
        expected_default = 1
        expected_type = int
        expected_doc = None

        arguments = self.methods[method_name]
        first_argument = arguments[0]
        argument_name, is_kwarg, argument_default, argument_type, _ = first_argument
        assert argument_name == expected_name
        assert is_kwarg is True
        assert argument_default == expected_default
        assert argument_type == expected_type

    def test_project_method_with_bool_default(self):

        method_name = 'kwarg_with_implicit_bool_type'
        expected_name = 'argument'
        expected_default = True
        expected_type = bool
        expected_doc = None

        arguments = self.methods[method_name]
        first_argument = arguments[0]
        argument_name, is_kwarg, argument_default, argument_type, _ = first_argument
        assert argument_name == expected_name
        assert is_kwarg is True
        assert argument_default == expected_default
        assert argument_type == expected_type


class TestProject:
    """Tests instantiating a `Project`."""

    def test_instantiating_project(self):
        instance = InstantiableProject()

    def test_instantiating_project_without_variables_fails(self):
        try:
            instance = NonInstantiableProject()
        except Exception:
            pass
        else:
            raise AssertionError(
                "Instantiating did not raise exception when it should have")
