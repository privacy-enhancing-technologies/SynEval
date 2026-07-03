"""Integration tests for CLI commands."""
import pytest
import os
import sys
from pathlib import Path
from click.testing import CliRunner
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cli.generate import cli


@pytest.fixture
def sample_data_path():
    """Get path to sample data fixture."""
    return Path(__file__).parent.parent / "fixtures" / "sample_data.csv"


@pytest.fixture
def temp_output(tmp_path):
    """Get temporary output path."""
    return tmp_path / "output.csv"


@pytest.fixture
def runner():
    """Create CLI runner."""
    return CliRunner()


class TestTiltedCLI:
    """Test Tilted generator CLI (simplest, no API calls)."""
    
    def test_tilted_basic(self, runner, sample_data_path, temp_output):
        """Test basic tilted generation."""
        result = runner.invoke(cli, [
            'tilted',
            '--input', str(sample_data_path),
            '--output', str(temp_output),
            '--text-column', 'use',
            '--tabular-column', 'sector',
            '--tabular-column', 'loan_amount',
            '--n-samples', '5'
        ])
        
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert temp_output.exists(), "Output file not created"
        
        # Verify output
        df = pd.read_csv(temp_output)
        assert len(df) == 5
        assert 'use' in df.columns
        assert 'sector' in df.columns
        assert 'loan_amount' in df.columns
    
    def test_tilted_with_strategy(self, runner, sample_data_path, temp_output):
        """Test tilted with different shuffle strategy."""
        result = runner.invoke(cli, [
            'tilted',
            '--input', str(sample_data_path),
            '--output', str(temp_output),
            '--text-column', 'use',
            '--tabular-column', 'sector',
            '--n-samples', '5',
            '--shuffle-strategy', 'random'
        ])
        
        assert result.exit_code == 0
        assert temp_output.exists()
    
    def test_tilted_missing_input(self, runner, temp_output):
        """Test error handling for missing input."""
        result = runner.invoke(cli, [
            'tilted',
            '--input', 'nonexistent.csv',
            '--output', str(temp_output),
            '--text-column', 'use',
            '--tabular-column', 'sector',
            '--n-samples', '5'
        ])
        
        assert result.exit_code != 0
    
    def test_tilted_missing_column(self, runner, sample_data_path, temp_output):
        """Test error handling for missing column."""
        result = runner.invoke(cli, [
            'tilted',
            '--input', str(sample_data_path),
            '--output', str(temp_output),
            '--text-column', 'nonexistent_column',
            '--tabular-column', 'sector',
            '--n-samples', '5'
        ])
        
        assert result.exit_code != 0


class TestConfigLoading:
    """Test configuration file loading."""
    
    def test_tilted_with_config(self, runner, sample_data_path, temp_output, tmp_path):
        """Test using config file."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("""
method: tilted
shuffle_strategy: random
random_state: 42
""")
        
        result = runner.invoke(cli, [
            'tilted',
            '--input', str(sample_data_path),
            '--output', str(temp_output),
            '--text-column', 'use',
            '--tabular-column', 'sector',
            '--n-samples', '5',
            '--config', str(config_file)
        ])
        
        assert result.exit_code == 0
        assert temp_output.exists()


class TestVerboseLogging:
    """Test verbose logging."""
    
    def test_verbose_flag(self, runner, sample_data_path, temp_output):
        """Test verbose logging flag."""
        result = runner.invoke(cli, [
            'tilted',
            '--input', str(sample_data_path),
            '--output', str(temp_output),
            '--text-column', 'use',
            '--tabular-column', 'sector',
            '--n-samples', '5',
            '--verbose'
        ])
        
        assert result.exit_code == 0
        # Verbose mode should complete successfully
        assert temp_output.exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
