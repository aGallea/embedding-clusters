from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestGetClusterName:
    def test_returns_short_name(self) -> None:
        from embedding_cluster.ai_naming import get_cluster_name

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Athletic Footwear"
        mock_response.choices = [mock_choice]

        with patch(
            "embedding_cluster.ai_naming.litellm_completion",
            return_value=mock_response,
        ):
            result = get_cluster_name(
                item_names=["Running Shoes", "Basketball Sneakers"],
                api_key="test-key",
                model="gpt-4o-mini",
            )

        assert result == "Athletic Footwear"

    def test_truncates_long_name(self) -> None:
        from embedding_cluster.ai_naming import get_cluster_name

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "A" * 50
        mock_response.choices = [mock_choice]

        with patch(
            "embedding_cluster.ai_naming.litellm_completion",
            return_value=mock_response,
        ):
            result = get_cluster_name(
                item_names=["item1"],
                api_key="test-key",
                model="gpt-4o-mini",
            )

        assert len(result) == 32  # 30 chars + ".."

    def test_none_content_returns_empty(self) -> None:
        from embedding_cluster.ai_naming import get_cluster_name

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = None
        mock_response.choices = [mock_choice]

        with patch(
            "embedding_cluster.ai_naming.litellm_completion",
            return_value=mock_response,
        ):
            result = get_cluster_name(
                item_names=["item1"],
                api_key="test-key",
                model="gpt-4o-mini",
            )

        assert result == ""

    def test_passes_base_url(self) -> None:
        from embedding_cluster.ai_naming import get_cluster_name

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Name"
        mock_response.choices = [mock_choice]

        with patch(
            "embedding_cluster.ai_naming.litellm_completion",
            return_value=mock_response,
        ) as mock_completion:
            get_cluster_name(
                item_names=["item1"],
                api_key="test-key",
                model="gpt-4o-mini",
                base_url="http://localhost:11434",
            )

        mock_completion.assert_called_once()
        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["api_base"] == "http://localhost:11434"

    def test_passes_temperature(self) -> None:
        from embedding_cluster.ai_naming import get_cluster_name

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Name"
        mock_response.choices = [mock_choice]

        with patch(
            "embedding_cluster.ai_naming.litellm_completion",
            return_value=mock_response,
        ) as mock_completion:
            get_cluster_name(
                item_names=["item1"],
                api_key="test-key",
                model="gpt-4o-mini",
                temperature=0.7,
            )

        mock_completion.assert_called_once()
        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["temperature"] == 0.7


class TestGetSubClusterName:
    def test_includes_parent_context(self) -> None:
        from embedding_cluster.ai_naming import get_sub_cluster_name

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Running Shoes"
        mock_response.choices = [mock_choice]

        with patch(
            "embedding_cluster.ai_naming.litellm_completion",
            return_value=mock_response,
        ) as mock_completion:
            result = get_sub_cluster_name(
                item_names=["Nike Air Max", "Adidas Ultraboost"],
                api_key="test-key",
                model="gpt-4o-mini",
                parent_cluster_name="Athletic Footwear",
            )

        assert result == "Running Shoes"
        call_kwargs = mock_completion.call_args[1]
        system_msg = call_kwargs["messages"][0]["content"]
        assert "Athletic Footwear" in system_msg

    def test_without_parent_name_uses_default_prompt(self) -> None:
        from embedding_cluster.ai_naming import get_sub_cluster_name

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Sub Name"
        mock_response.choices = [mock_choice]

        with patch(
            "embedding_cluster.ai_naming.litellm_completion",
            return_value=mock_response,
        ) as mock_completion:
            result = get_sub_cluster_name(
                item_names=["item1"],
                api_key="test-key",
                model="gpt-4o-mini",
            )

        assert result == "Sub Name"
        call_kwargs = mock_completion.call_args[1]
        system_msg = call_kwargs["messages"][0]["content"]
        assert "sub-group" not in system_msg


class TestTestConnection:
    def test_success(self) -> None:
        from embedding_cluster.ai_naming import test_connection

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Hello"
        mock_response.choices = [mock_choice]

        with patch(
            "embedding_cluster.ai_naming.litellm_completion",
            return_value=mock_response,
        ):
            success, error = test_connection(
                api_key="test-key",
                model="gpt-4o-mini",
            )

        assert success is True
        assert error is None

    def test_failure_redacts_key(self) -> None:
        from embedding_cluster.ai_naming import test_connection

        with patch(
            "embedding_cluster.ai_naming.litellm_completion",
            side_effect=Exception("Invalid API key: sk-1234567890abcdef"),
        ):
            success, error = test_connection(
                api_key="sk-1234567890abcdef",
                model="gpt-4o-mini",
            )

        assert success is False
        assert error is not None
        assert "sk-1234567890abcdef" not in error
