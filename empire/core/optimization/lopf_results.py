"""
Load Flow Results Module

This module provides comprehensive output and logging for DC-OPF/LOPF results,
enabling detailed evaluation of load flow outcomes including bus angles,
line flows, reactances, susceptances, voltage angles, and constraint violations.
"""
from __future__ import annotations
import csv
import logging
from pathlib import Path
from pyomo.environ import value
from typing import Optional

logger = logging.getLogger(__name__)


def write_angle_based_results(
    instance,
    result_file_path: Path,
    logger_inst: Optional[logging.Logger] = None
) -> None:
    """
    Write detailed angle-based DC-OPF results to CSV files.
    
    Outputs:
    - Bus angles (Theta) for all nodes, hours, scenarios, periods
    - DC flows (FlowDC) for all directional links
    - Reactances and susceptances per line
    - Ohm's law validation (actual vs theoretical flow)
    - Line loadings (% of capacity)
    - Angle differences across lines
    
    Args:
        instance: Solved Pyomo model instance
        result_file_path: Directory path for output files
        logger_inst: Optional logger instance (defaults to module logger)
    """
    if logger_inst is None:
        logger_inst = logger
        
    # Check if angle-based formulation was used
    if not hasattr(instance, 'Theta'):
        logger_inst.info("Angle-based DC-OPF not detected (no Theta variable). Skipping angle results.")
        return
        
    logger_inst.info("Writing angle-based DC-OPF results...")
    
    # Helper to get investment period names
    inv_per = _get_investment_periods(instance)
    
    # ========================================
    # 1. Bus Angles (degrees and radians)
    # ========================================
    angles_file = result_file_path / 'results_lopf_bus_angles.csv'
    with open(angles_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Node", "Period", "Scenario", "Season", "Hour",
            "Angle_rad", "Angle_deg"
        ])
        
        for n in instance.Node:
            for i in instance.PeriodActive:
                for w in instance.Scenario:
                    for (s, h) in instance.HoursOfSeason:
                        theta_val = value(instance.Theta[n, h, w, i])
                        writer.writerow([
                            n,
                            inv_per[int(i-1)],
                            w,
                            s,
                            h,
                            theta_val,
                            theta_val * 180.0 / 3.14159  # Convert to degrees
                        ])
    
    logger_inst.info(f"  [OK] Bus angles written to {angles_file.name}")
    
    # ========================================
    # 2. DC Flows with Line Parameters
    # ========================================
    flows_file = result_file_path / 'results_lopf_dc_flows.csv'
    with open(flows_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "FromNode", "ToNode", "Period", "Scenario", "Season", "Hour",
            "FlowDC_MW", "Capacity_MW", "Loading_pct",
            "Reactance_pu", "Susceptance_pu",
            "AngleDiff_rad", "AngleDiff_deg",
            "TheoreticalFlow_MW", "FlowError_MW", "FlowError_pct"
        ])
        
        for (i_node, j_node) in instance.DirectionalLink:
            for i_per in instance.PeriodActive:
                # Get capacity (try different possible names)
                try:
                    cap = value(instance.CapacityDir[i_node, j_node, i_per])
                except:
                    try:
                        cap = value(instance.transmissionInstalledCap[i_node, j_node, i_per])
                    except:
                        cap = 0.0
                
                # Get reactance
                try:
                    reactance = value(instance._reactance_dir[i_node, j_node])
                except:
                    reactance = None
                    
                for w in instance.Scenario:
                    for (s, h) in instance.HoursOfSeason:
                        flow = value(instance.FlowDC[i_node, j_node, h, w, i_per])
                        theta_i = value(instance.Theta[i_node, h, w, i_per])
                        theta_j = value(instance.Theta[j_node, h, w, i_per])
                        angle_diff = theta_i - theta_j
                        
                        # Calculate susceptance and theoretical flow
                        if reactance is not None and abs(reactance) > 1e-9:
                            susceptance = 1.0 / reactance
                            theoretical_flow = susceptance * angle_diff
                            flow_error = abs(flow - theoretical_flow)
                            flow_error_pct = (flow_error / (abs(theoretical_flow) + 1e-6)) * 100.0
                        else:
                            susceptance = None
                            theoretical_flow = None
                            flow_error = None
                            flow_error_pct = None
                        
                        # Calculate loading percentage
                        loading_pct = (abs(flow) / cap * 100.0) if cap > 1e-6 else 0.0
                        
                        writer.writerow([
                            i_node,
                            j_node,
                            inv_per[int(i_per-1)],
                            w,
                            s,
                            h,
                            flow,
                            cap,
                            loading_pct,
                            reactance if reactance is not None else 'N/A',
                            susceptance if susceptance is not None else 'N/A',
                            angle_diff,
                            angle_diff * 180.0 / 3.14159,
                            theoretical_flow if theoretical_flow is not None else 'N/A',
                            flow_error if flow_error is not None else 'N/A',
                            flow_error_pct if flow_error_pct is not None else 'N/A'
                        ])
    
    logger_inst.info(f"  [OK] DC flows and line parameters written to {flows_file.name}")
    
    # ========================================
    # 3. Summary Statistics per Period
    # ========================================
    summary_file = result_file_path / 'results_lopf_summary.csv'
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Period", "Scenario",
            "MaxAngle_deg", "MinAngle_deg", "AngleRange_deg",
            "MaxLoading_pct", "AvgLoading_pct",
            "MaxFlowError_MW", "AvgFlowError_MW",
            "NumCongestedLines", "NumActiveLines"
        ])
        
        for i_per in instance.PeriodActive:
            for w in instance.Scenario:
                # Collect statistics across all hours
                all_angles = []
                all_loadings = []
                all_errors = []
                congested_count = 0
                active_count = 0
                
                for n in instance.Node:
                    for (s, h) in instance.HoursOfSeason:
                        theta_val = value(instance.Theta[n, h, w, i_per])
                        all_angles.append(theta_val * 180.0 / 3.14159)
                
                for (i_node, j_node) in instance.DirectionalLink:
                    try:
                        cap = value(instance.CapacityDir[i_node, j_node, i_per])
                    except:
                        try:
                            cap = value(instance.transmissionInstalledCap[i_node, j_node, i_per])
                        except:
                            cap = 0.0
                    
                    try:
                        reactance = value(instance._reactance_dir[i_node, j_node])
                    except:
                        reactance = None
                    
                    for (s, h) in instance.HoursOfSeason:
                        flow = value(instance.FlowDC[i_node, j_node, h, w, i_per])
                        
                        if abs(flow) > 1e-6:
                            active_count += 1
                        
                        if cap > 1e-6:
                            loading = abs(flow) / cap * 100.0
                            all_loadings.append(loading)
                            if loading > 95.0:  # Consider >95% as congested
                                congested_count += 1
                        
                        if reactance is not None and abs(reactance) > 1e-9:
                            theta_i = value(instance.Theta[i_node, h, w, i_per])
                            theta_j = value(instance.Theta[j_node, h, w, i_per])
                            angle_diff = theta_i - theta_j
                            susceptance = 1.0 / reactance
                            theoretical_flow = susceptance * angle_diff
                            flow_error = abs(flow - theoretical_flow)
                            all_errors.append(flow_error)
                
                writer.writerow([
                    inv_per[int(i_per-1)],
                    w,
                    max(all_angles) if all_angles else 0.0,
                    min(all_angles) if all_angles else 0.0,
                    (max(all_angles) - min(all_angles)) if all_angles else 0.0,
                    max(all_loadings) if all_loadings else 0.0,
                    sum(all_loadings) / len(all_loadings) if all_loadings else 0.0,
                    max(all_errors) if all_errors else 0.0,
                    sum(all_errors) / len(all_errors) if all_errors else 0.0,
                    congested_count,
                    active_count
                ])
    
    logger_inst.info(f"  [OK] Summary statistics written to {summary_file.name}")
    
    # ========================================
    # 4. Line Reactance and Susceptance Table
    # ========================================
    params_file = result_file_path / 'results_lopf_line_parameters.csv'
    with open(params_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "FromNode", "ToNode",
            "Reactance_pu", "Susceptance_pu",
            "IsCandidate", "IsExisting"
        ])
        
        for (i_node, j_node) in instance.DirectionalLink:
            try:
                reactance = value(instance._reactance_dir[i_node, j_node])
                susceptance = 1.0 / reactance if abs(reactance) > 1e-9 else None
            except:
                reactance = None
                susceptance = None
            
            is_candidate = (i_node, j_node) in instance.CandidateTransmission if hasattr(instance, 'CandidateTransmission') else False
            is_existing = (i_node, j_node) in instance.ExistingTransmission if hasattr(instance, 'ExistingTransmission') else False
            
            writer.writerow([
                i_node,
                j_node,
                reactance if reactance is not None else 'N/A',
                susceptance if susceptance is not None else 'N/A',
                'Yes' if is_candidate else 'No',
                'Yes' if is_existing else 'No'
            ])
    
    logger_inst.info(f"  [OK] Line parameters written to {params_file.name}")
    
    logger_inst.info("Angle-based DC-OPF results written successfully.")


def log_lopf_diagnostics(instance, logger_inst: Optional[logging.Logger] = None) -> None:
    """
    Log detailed diagnostics about the LOPF formulation and solution.
    
    This function logs at INFO and DEBUG levels to provide insights into:
    - Formulation type (Kirchhoff, Angle, PTDF)
    - Number of constraints and variables
    - Angle bounds and reference node
    - Line loading statistics
    - Potential numerical issues
    
    Args:
        instance: Solved Pyomo model instance
        logger_inst: Optional logger instance (defaults to module logger)
    """
    if logger_inst is None:
        logger_inst = logger
    
    logger_inst.info("=" * 70)
    logger_inst.info("LOAD FLOW DIAGNOSTIC REPORT")
    logger_inst.info("=" * 70)
    
    # Detect formulation type
    formulation = "Unknown"
    if hasattr(instance, 'Theta'):
        formulation = "Angle-based DC-OPF"
    elif hasattr(instance, 'FlowK'):
        formulation = "Kirchhoff Cycle-based DC-OPF"
    elif hasattr(instance, 'PTDF'):
        formulation = "PTDF-based DC-OPF"
    
    logger_inst.info(f"Formulation: {formulation}")
    
    # Count components
    if hasattr(instance, 'DirectionalLink'):
        n_arcs = len(instance.DirectionalLink)
        logger_inst.info(f"Directional Links: {n_arcs}")
    
    if hasattr(instance, 'Node'):
        n_nodes = len(instance.Node)
        logger_inst.info(f"Nodes: {n_nodes}")
    
    if hasattr(instance, 'CandidateTransmission'):
        n_cand = len(instance.CandidateTransmission)
        logger_inst.info(f"Candidate Lines: {n_cand}")
    
    if hasattr(instance, 'ExistingTransmission'):
        n_exist = len(instance.ExistingTransmission)
        logger_inst.info(f"Existing Lines: {n_exist}")
    
    # Angle-based specific diagnostics
    if hasattr(instance, 'Theta'):
        logger_inst.info("-" * 70)
        logger_inst.info("ANGLE-BASED DC-OPF DIAGNOSTICS")
        logger_inst.info("-" * 70)
        
        # Angle bounds
        if hasattr(instance, 'AngleMax'):
            angle_max_rad = value(instance.AngleMax)
            angle_max_deg = angle_max_rad * 180.0 / 3.14159
            logger_inst.info(f"Angle Bounds: ±{angle_max_deg:.2f}° (±{angle_max_rad:.4f} rad)")
        
        # Find reference node
        if hasattr(instance, 'AngleRef'):
            logger_inst.info("Angle Reference: Fixed (slack node constraint active)")
        else:
            logger_inst.info("Angle Reference: Not fixed (may have multiple solutions)")
        
        # Sample some angles (first period, first scenario, first hour)
        if len(instance.PeriodActive) > 0 and len(instance.Scenario) > 0:
            i_sample = list(instance.PeriodActive)[0]
            w_sample = list(instance.Scenario)[0]
            h_sample = list(instance.HoursOfSeason)[0][1]
            
            logger_inst.info(f"\nSample Bus Angles (Period {i_sample}, Scenario {w_sample}, Hour {h_sample}):")
            node_count = 0
            for n in instance.Node:
                if node_count < 10:  # Show first 10 nodes
                    theta_val = value(instance.Theta[n, h_sample, w_sample, i_sample])
                    theta_deg = theta_val * 180.0 / 3.14159
                    logger_inst.info(f"  {n}: {theta_deg:8.4f}° ({theta_val:8.6f} rad)")
                    node_count += 1
            if len(instance.Node) > 10:
                logger_inst.info(f"  ... ({len(instance.Node) - 10} more nodes)")
        
        # Flow statistics
        if hasattr(instance, 'FlowDC'):
            logger_inst.info("\nDC Flow Statistics:")
            max_flow = 0.0
            max_flow_arc = None
            total_flow = 0.0
            n_flows = 0
            
            for (i_node, j_node) in instance.DirectionalLink:
                for i_per in instance.PeriodActive:
                    for w in instance.Scenario:
                        for (s, h) in instance.HoursOfSeason:
                            flow = abs(value(instance.FlowDC[i_node, j_node, h, w, i_per]))
                            total_flow += flow
                            n_flows += 1
                            if flow > max_flow:
                                max_flow = flow
                                max_flow_arc = (i_node, j_node, i_per, w, s, h)
            
            avg_flow = total_flow / n_flows if n_flows > 0 else 0.0
            logger_inst.info(f"  Maximum Flow: {max_flow:.2f} MW")
            if max_flow_arc:
                logger_inst.info(f"    at arc {max_flow_arc[0]} -> {max_flow_arc[1]}, Period {max_flow_arc[2]}, Scenario {max_flow_arc[3]}")
            logger_inst.info(f"  Average Flow: {avg_flow:.2f} MW")
        
        # Check for constraint violations (if dual variables available)
        if hasattr(instance, 'OhmLawDC_Exist'):
            logger_inst.info("\nOhm's Law Constraints: Active for existing lines")
        if hasattr(instance, 'OhmLawDC_Cand_UB') and hasattr(instance, 'OhmLawDC_Cand_LB'):
            logger_inst.info("Candidate Line Constraints: Big-M formulation active")
    
    # Kirchhoff-specific diagnostics
    elif hasattr(instance, 'FlowK'):
        logger_inst.info("-" * 70)
        logger_inst.info("KIRCHHOFF CYCLE-BASED DC-OPF DIAGNOSTICS")
        logger_inst.info("-" * 70)
        
        if hasattr(instance, 'Cycle'):
            n_cycles = len(instance.Cycle)
            logger_inst.info(f"Fundamental Cycles: {n_cycles}")
        
        if hasattr(instance, 'KVL'):
            logger_inst.info("KVL Constraints: Active on all cycles")
    
    logger_inst.info("=" * 70)
    logger_inst.info("END OF DIAGNOSTIC REPORT")
    logger_inst.info("=" * 70)


def _get_investment_periods(instance):
    """Helper to get investment period names (e.g., '2025', '2030', etc.)"""
    try:
        # Common pattern in EMPIRE models
        base_year = 2020
        leap_years = value(instance.LeapYearsInvestment)
        return [str(base_year + (i) * leap_years) for i in instance.PeriodActive]
    except:
        # Fallback: just use period indices
        return [str(i) for i in instance.PeriodActive]
