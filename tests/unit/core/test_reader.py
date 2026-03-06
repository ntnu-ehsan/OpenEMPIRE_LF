"""
Unit tests for empire.core.reader module
"""
import pytest
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from empire.core.reader import read_bidirectional_to_directional


class TestBidirectionalToDirectional:
    """Test the bidirectional to directional expansion for line parameters"""
    
    def test_basic_expansion(self, tmp_path):
        """Test basic bidirectional to directional expansion"""
        # Create sample bidirectional data
        data = {
            'FromNode': ['A', 'B', 'C'],
            'ToNode': ['B', 'C', 'D'],
            'lineReactance': [0.001, 0.002, 0.003]
        }
        df = pd.DataFrame(data)
        
        # Create a mock Excel file
        excel_path = tmp_path / "test_transmission.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Add header rows
            header_df = pd.DataFrame([['Line Reactance Values'], 
                                     ['FromNode', 'ToNode', 'lineReactance']])
            header_df.to_excel(writer, sheet_name='lineReactance', index=False, header=False)
            # Add data starting from row 2 (0-indexed)
            df.to_excel(writer, sheet_name='lineReactance', startrow=2, index=False, header=False)
        
        # Read and process
        excel_data = pd.read_excel(excel_path, sheet_name=None)
        tab_path = tmp_path / "tabs"
        read_bidirectional_to_directional(excel_data, 'lineReactance', [0, 1, 2], 
                                         tab_path, "Transmission", skipheaders=2)
        
        # Read the generated tab file
        tab_file = tab_path / "Transmission_lineReactance.tab"
        assert tab_file.exists(), "Tab file should be created"
        
        result_df = pd.read_csv(tab_file, sep='\t')
        
        # Should have 6 rows (3 original + 3 reversed)
        assert len(result_df) == 6, f"Expected 6 rows, got {len(result_df)}"
        
        # Check that both directions exist for each pair
        expected_pairs = [
            ('A', 'B', 0.001), ('B', 'A', 0.001),
            ('B', 'C', 0.002), ('C', 'B', 0.002),
            ('C', 'D', 0.003), ('D', 'C', 0.003)
        ]
        
        for from_node, to_node, reactance in expected_pairs:
            matching_rows = result_df[
                (result_df['FromNode'] == from_node) & 
                (result_df['ToNode'] == to_node)
            ]
            assert len(matching_rows) == 1, f"Expected one row for {from_node}->{to_node}"
            assert matching_rows.iloc[0]['lineReactance'] == reactance, \
                f"Expected reactance {reactance} for {from_node}->{to_node}"
    
    def test_empty_data(self, tmp_path):
        """Test handling of empty sheet"""
        # Create empty Excel file with just headers
        excel_path = tmp_path / "test_transmission_empty.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            header_df = pd.DataFrame([['Line Reactance Values'], 
                                     ['FromNode', 'ToNode', 'lineReactance']])
            header_df.to_excel(writer, sheet_name='lineReactance', index=False, header=False)
        
        excel_data = pd.read_excel(excel_path, sheet_name=None)
        tab_path = tmp_path / "tabs_empty"
        read_bidirectional_to_directional(excel_data, 'lineReactance', [0, 1, 2], 
                                         tab_path, "Transmission", skipheaders=2)
        
        tab_file = tab_path / "Transmission_lineReactance.tab"
        assert tab_file.exists(), "Tab file should be created even for empty data"
        
        result_df = pd.read_csv(tab_file, sep='\t')
        # Should be empty (only headers)
        assert len(result_df) == 0, "Empty input should produce empty output"
    
    def test_whitespace_removal(self, tmp_path):
        """Test that whitespace is removed from node names"""
        data = {
            'FromNode': ['A ', ' B', ' C '],
            'ToNode': [' B', 'C ', ' D'],
            'lineReactance': [0.001, 0.002, 0.003]
        }
        df = pd.DataFrame(data)
        
        excel_path = tmp_path / "test_transmission_ws.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            header_df = pd.DataFrame([['Line Reactance Values'], 
                                     ['FromNode', 'ToNode', 'lineReactance']])
            header_df.to_excel(writer, sheet_name='lineReactance', index=False, header=False)
            df.to_excel(writer, sheet_name='lineReactance', startrow=2, index=False, header=False)
        
        excel_data = pd.read_excel(excel_path, sheet_name=None)
        tab_path = tmp_path / "tabs_ws"
        read_bidirectional_to_directional(excel_data, 'lineReactance', [0, 1, 2], 
                                         tab_path, "Transmission", skipheaders=2)
        
        tab_file = tab_path / "Transmission_lineReactance.tab"
        result_df = pd.read_csv(tab_file, sep='\t')
        
        # Check that whitespace is removed
        for col in ['FromNode', 'ToNode']:
            assert not any(result_df[col].str.contains(r'\s', regex=True).fillna(False)), \
                f"Whitespace should be removed from {col}"
    
    def test_symmetric_reactance(self, tmp_path):
        """Test that reactance values are symmetric in both directions"""
        data = {
            'FromNode': ['Norway', 'Sweden'],
            'ToNode': ['Sweden', 'Finland'],
            'lineReactance': [0.0045, 0.0038]
        }
        df = pd.DataFrame(data)
        
        excel_path = tmp_path / "test_transmission_sym.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            header_df = pd.DataFrame([['Line Reactance Values'], 
                                     ['FromNode', 'ToNode', 'lineReactance']])
            header_df.to_excel(writer, sheet_name='lineReactance', index=False, header=False)
            df.to_excel(writer, sheet_name='lineReactance', startrow=2, index=False, header=False)
        
        excel_data = pd.read_excel(excel_path, sheet_name=None)
        tab_path = tmp_path / "tabs_sym"
        read_bidirectional_to_directional(excel_data, 'lineReactance', [0, 1, 2], 
                                         tab_path, "Transmission", skipheaders=2)
        
        tab_file = tab_path / "Transmission_lineReactance.tab"
        result_df = pd.read_csv(tab_file, sep='\t')
        
        # Check Norway->Sweden and Sweden->Norway have same reactance
        no_se = result_df[(result_df['FromNode'] == 'Norway') & (result_df['ToNode'] == 'Sweden')]
        se_no = result_df[(result_df['FromNode'] == 'Sweden') & (result_df['ToNode'] == 'Norway')]
        
        assert len(no_se) == 1 and len(se_no) == 1, "Both directions should exist"
        assert no_se.iloc[0]['lineReactance'] == se_no.iloc[0]['lineReactance'] == 0.0045, \
            "Reactance should be symmetric"
        
        # Check Sweden->Finland and Finland->Sweden
        se_fi = result_df[(result_df['FromNode'] == 'Sweden') & (result_df['ToNode'] == 'Finland')]
        fi_se = result_df[(result_df['FromNode'] == 'Finland') & (result_df['ToNode'] == 'Sweden')]
        
        assert len(se_fi) == 1 and len(fi_se) == 1, "Both directions should exist"
        assert se_fi.iloc[0]['lineReactance'] == fi_se.iloc[0]['lineReactance'] == 0.0038, \
            "Reactance should be symmetric"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
