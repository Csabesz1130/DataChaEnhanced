"""
Formula Learning Module

This module provides advanced capabilities for learning and understanding Excel formula logic,
including calculations, filtering operations, and conditional logic. It goes beyond just storing
formula strings to actually understanding the underlying operations and being able to apply
similar logic to new data or different contexts.
"""

import re
import ast
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FormulaType(Enum):
    """Types of Excel formulas"""
    SUM = "SUM"
    AVERAGE = "AVERAGE"
    COUNT = "COUNT"
    MAX = "MAX"
    MIN = "MIN"
    IF = "IF"
    VLOOKUP = "VLOOKUP"
    HLOOKUP = "HLOOKUP"
    FILTER = "FILTER"
    SORT = "SORT"
    CONCATENATE = "CONCATENATE"
    DATE = "DATE"
    MATH = "MATH"
    LOGICAL = "LOGICAL"
    UNKNOWN = "UNKNOWN"


@dataclass
class FormulaLogic:
    """Represents the learned logic of an Excel formula"""
    formula_type: FormulaType
    operation: str
    source_range: Optional[str] = None
    target_range: Optional[str] = None
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    result_type: str = "numeric"
    confidence: float = 1.0


@dataclass
class FilterCondition:
    """Represents a filtering condition"""
    column: str
    operator: str  # >, <, >=, <=, ==, !=, contains, starts_with, ends_with
    value: Any
    logical_operator: str = "AND"  # AND, OR


@dataclass
class CalculationPattern:
    """Represents a learned calculation pattern"""
    pattern_id: str
    description: str
    formula_logic: FormulaLogic
    examples: List[str] = field(default_factory=list)
    frequency: int = 1
    success_rate: float = 1.0


class FormulaLearner:
    """
    Advanced formula learning system that understands Excel formula logic
    and can apply learned patterns to new data
    """
    
    def __init__(self):
        self.learned_patterns: Dict[str, CalculationPattern] = {}
        self.formula_templates: Dict[str, str] = {}
        self.filter_patterns: List[FilterCondition] = []
        
    def learn_formula_logic(self, formula: str, context_data: pd.DataFrame = None) -> FormulaLogic:
        """
        Learn the logic behind an Excel formula
        
        Args:
            formula: The Excel formula string
            context_data: Optional DataFrame for context analysis
            
        Returns:
            FormulaLogic object representing the learned logic
        """
        try:
            # Clean and normalize formula
            clean_formula = self._normalize_formula(formula)
            
            # Identify formula type
            formula_type = self._identify_formula_type(clean_formula)
            
            # Extract operation details
            operation_details = self._extract_operation_details(clean_formula, formula_type)
            
            # Analyze dependencies
            dependencies = self._extract_dependencies(clean_formula)
            
            # Create FormulaLogic object
            formula_logic = FormulaLogic(
                formula_type=formula_type,
                operation=operation_details.get('operation', ''),
                source_range=operation_details.get('source_range'),
                target_range=operation_details.get('target_range'),
                conditions=operation_details.get('conditions', []),
                parameters=operation_details.get('parameters', {}),
                dependencies=dependencies,
                result_type=self._determine_result_type(formula_type, operation_details)
            )
            
            # Store pattern for future use
            self._store_pattern(formula_logic, formula)
            
            return formula_logic
            
        except Exception as e:
            logger.error(f"Error learning formula logic: {e}")
            return FormulaLogic(
                formula_type=FormulaType.UNKNOWN,
                operation="unknown",
                confidence=0.0
            )
    
    def _normalize_formula(self, formula: str) -> str:
        """Normalize formula for consistent parsing"""
        # Remove leading '='
        if formula.startswith('='):
            formula = formula[1:]
        
        # Normalize whitespace
        formula = re.sub(r'\s+', ' ', formula.strip())
        
        return formula.upper()
    
    def _identify_formula_type(self, formula: str) -> FormulaType:
        """Identify the type of Excel formula"""
        if re.match(r'^SUM\s*\(', formula):
            return FormulaType.SUM
        elif re.match(r'^AVERAGE\s*\(', formula):
            return FormulaType.AVERAGE
        elif re.match(r'^COUNT\s*\(', formula):
            return FormulaType.COUNT
        elif re.match(r'^MAX\s*\(', formula):
            return FormulaType.MAX
        elif re.match(r'^MIN\s*\(', formula):
            return FormulaType.MIN
        elif re.match(r'^IF\s*\(', formula):
            return FormulaType.IF
        elif re.match(r'^VLOOKUP\s*\(', formula):
            return FormulaType.VLOOKUP
        elif re.match(r'^HLOOKUP\s*\(', formula):
            return FormulaType.HLOOKUP
        elif re.match(r'^FILTER\s*\(', formula):
            return FormulaType.FILTER
        elif re.match(r'^SORT\s*\(', formula):
            return FormulaType.SORT
        elif re.match(r'^CONCATENATE\s*\(', formula):
            return FormulaType.CONCATENATE
        elif re.search(r'[+\-*/^]', formula):
            return FormulaType.MATH
        elif re.search(r'(AND|OR|NOT|XOR)\s*\(', formula):
            return FormulaType.LOGICAL
        else:
            return FormulaType.UNKNOWN
    
    def _extract_operation_details(self, formula: str, formula_type: FormulaType) -> Dict[str, Any]:
        """Extract detailed operation information from formula"""
        details = {
            'operation': formula_type.value,
            'source_range': None,
            'target_range': None,
            'conditions': [],
            'parameters': {}
        }
        
        if formula_type == FormulaType.SUM:
            details.update(self._extract_sum_details(formula))
        elif formula_type == FormulaType.AVERAGE:
            details.update(self._extract_average_details(formula))
        elif formula_type == FormulaType.IF:
            details.update(self._extract_if_details(formula))
        elif formula_type == FormulaType.FILTER:
            details.update(self._extract_filter_details(formula))
        elif formula_type == FormulaType.MATH:
            details.update(self._extract_math_details(formula))
        
        return details
    
    def _extract_sum_details(self, formula: str) -> Dict[str, Any]:
        """Extract details from SUM formula"""
        # Extract range from SUM(A1:A10) or SUM(A1,A2,A3)
        range_match = re.search(r'SUM\s*\(([^)]+)\)', formula)
        if range_match:
            range_str = range_match.group(1)
            return {
                'source_range': range_str,
                'parameters': {'range': range_str}
            }
        return {}
    
    def _extract_average_details(self, formula: str) -> Dict[str, Any]:
        """Extract details from AVERAGE formula"""
        range_match = re.search(r'AVERAGE\s*\(([^)]+)\)', formula)
        if range_match:
            range_str = range_match.group(1)
            return {
                'source_range': range_str,
                'parameters': {'range': range_str}
            }
        return {}
    
    def _extract_if_details(self, formula: str) -> Dict[str, Any]:
        """Extract details from IF formula"""
        # IF(condition, value_if_true, value_if_false)
        if_match = re.search(r'IF\s*\(([^)]+)\)', formula)
        if if_match:
            params = if_match.group(1).split(',')
            if len(params) >= 3:
                condition = params[0].strip()
                true_value = params[1].strip()
                false_value = params[2].strip()
                
                # Parse condition
                condition_details = self._parse_condition(condition)
                
                return {
                    'conditions': [condition_details],
                    'parameters': {
                        'true_value': true_value,
                        'false_value': false_value,
                        'condition': condition
                    }
                }
        return {}
    
    def _extract_filter_details(self, formula: str) -> Dict[str, Any]:
        """Extract details from FILTER formula"""
        filter_match = re.search(r'FILTER\s*\(([^)]+)\)', formula)
        if filter_match:
            params = filter_match.group(1).split(',')
            if len(params) >= 2:
                array = params[0].strip()
                include = params[1].strip()
                
                # Parse include condition
                condition_details = self._parse_condition(include)
                
                return {
                    'source_range': array,
                    'conditions': [condition_details],
                    'parameters': {
                        'array': array,
                        'include': include
                    }
                }
        return {}
    
    def _extract_math_details(self, formula: str) -> Dict[str, Any]:
        """Extract details from mathematical formula"""
        # Extract mathematical operations and operands
        operations = re.findall(r'[+\-*/^]', formula)
        operands = re.findall(r'[A-Z]+\d+|[A-Z]+:[A-Z]+\d+|\d+\.?\d*', formula)
        
        return {
            'parameters': {
                'operations': operations,
                'operands': operands,
                'formula': formula
            }
        }
    
    def _parse_condition(self, condition: str) -> Dict[str, Any]:
        """Parse a condition string into structured format"""
        # Common condition patterns
        patterns = [
            (r'([A-Z]+\d+)\s*([><=!]+)\s*(.+)', 'cell_comparison'),
            (r'([A-Z]+)\s*([><=!]+)\s*(.+)', 'column_comparison'),
            (r'([A-Z]+\d+:[A-Z]+\d+)\s*([><=!]+)\s*(.+)', 'range_comparison')
        ]
        
        for pattern, condition_type in patterns:
            match = re.match(pattern, condition)
            if match:
                return {
                    'type': condition_type,
                    'left_operand': match.group(1),
                    'operator': match.group(2),
                    'right_operand': match.group(3).strip('"\'')
                }
        
        return {
            'type': 'unknown',
            'condition': condition
        }
    
    def _extract_dependencies(self, formula: str) -> List[str]:
        """Extract cell/range dependencies from formula"""
        # Find all cell references (A1, B2, etc.) and ranges (A1:B10)
        dependencies = re.findall(r'[A-Z]+\d+(?::[A-Z]+\d+)?', formula)
        return list(set(dependencies))
    
    def _determine_result_type(self, formula_type: FormulaType, details: Dict[str, Any]) -> str:
        """Determine the expected result type of the formula"""
        if formula_type in [FormulaType.SUM, FormulaType.AVERAGE, FormulaType.MAX, FormulaType.MIN]:
            return "numeric"
        elif formula_type == FormulaType.COUNT:
            return "integer"
        elif formula_type == FormulaType.IF:
            return "mixed"
        elif formula_type == FormulaType.FILTER:
            return "array"
        elif formula_type == FormulaType.CONCATENATE:
            return "text"
        else:
            return "unknown"
    
    def _store_pattern(self, formula_logic: FormulaLogic, original_formula: str):
        """Store learned pattern for future use"""
        pattern_id = f"{formula_logic.formula_type.value}_{len(self.learned_patterns)}"
        
        pattern = CalculationPattern(
            pattern_id=pattern_id,
            description=f"Learned {formula_logic.formula_type.value} pattern",
            formula_logic=formula_logic,
            examples=[original_formula]
        )
        
        self.learned_patterns[pattern_id] = pattern
    
    def apply_learned_logic(self, formula_logic: FormulaLogic, data: pd.DataFrame, 
                          target_context: str = None) -> Union[pd.Series, pd.DataFrame, Any]:
        """
        Apply learned formula logic to new data
        
        Args:
            formula_logic: The learned formula logic
            data: DataFrame to apply logic to
            target_context: Optional target context for the operation
            
        Returns:
            Result of applying the learned logic
        """
        try:
            if formula_logic.formula_type == FormulaType.SUM:
                return self._apply_sum_logic(formula_logic, data)
            elif formula_logic.formula_type == FormulaType.AVERAGE:
                return self._apply_average_logic(formula_logic, data)
            elif formula_logic.formula_type == FormulaType.IF:
                return self._apply_if_logic(formula_logic, data)
            elif formula_logic.formula_type == FormulaType.FILTER:
                return self._apply_filter_logic(formula_logic, data)
            elif formula_logic.formula_type == FormulaType.MATH:
                return self._apply_math_logic(formula_logic, data)
            else:
                logger.warning(f"Unsupported formula type: {formula_logic.formula_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error applying learned logic: {e}")
            return None
    
    def _apply_sum_logic(self, formula_logic: FormulaLogic, data: pd.DataFrame) -> float:
        """Apply SUM logic to data"""
        if formula_logic.source_range:
            # Parse range and extract values
            values = self._extract_values_from_range(data, formula_logic.source_range)
            return sum(values)
        return 0.0
    
    def _apply_average_logic(self, formula_logic: FormulaLogic, data: pd.DataFrame) -> float:
        """Apply AVERAGE logic to data"""
        if formula_logic.source_range:
            values = self._extract_values_from_range(data, formula_logic.source_range)
            return np.mean(values) if values else 0.0
        return 0.0
    
    def _apply_if_logic(self, formula_logic: FormulaLogic, data: pd.DataFrame) -> pd.Series:
        """Apply IF logic to data"""
        if not formula_logic.conditions:
            return pd.Series()
        
        condition = formula_logic.conditions[0]
        true_value = formula_logic.parameters.get('true_value', 0)
        false_value = formula_logic.parameters.get('false_value', 0)
        
        # Create condition mask
        mask = self._create_condition_mask(data, condition)
        
        # Apply IF logic
        result = pd.Series(false_value, index=data.index)
        result[mask] = true_value
        
        return result
    
    def _apply_filter_logic(self, formula_logic: FormulaLogic, data: pd.DataFrame) -> pd.DataFrame:
        """Apply FILTER logic to data"""
        if not formula_logic.conditions:
            return data
        
        condition = formula_logic.conditions[0]
        mask = self._create_condition_mask(data, condition)
        
        return data[mask]
    
    def _apply_math_logic(self, formula_logic: FormulaLogic, data: pd.DataFrame) -> pd.Series:
        """Apply mathematical logic to data"""
        formula = formula_logic.parameters.get('formula', '')
        operands = formula_logic.parameters.get('operands', [])
        
        # Create a safe evaluation environment
        safe_dict = {}
        for i, operand in enumerate(operands):
            if re.match(r'[A-Z]+\d+', operand):
                # Cell reference - extract value
                value = self._extract_cell_value(data, operand)
                safe_dict[f'x{i}'] = value
            elif re.match(r'\d+\.?\d*', operand):
                # Numeric value
                safe_dict[f'x{i}'] = float(operand)
        
        # Replace operands in formula with safe variables
        eval_formula = formula
        for i, operand in enumerate(operands):
            eval_formula = eval_formula.replace(operand, f'x{i}')
        
        try:
            # Safe evaluation (in production, use a more secure method)
            result = eval(eval_formula, {"__builtins__": {}}, safe_dict)
            return pd.Series(result, index=data.index)
        except:
            logger.warning(f"Could not evaluate formula: {formula}")
            return pd.Series(0, index=data.index)
    
    def _extract_values_from_range(self, data: pd.DataFrame, range_str: str) -> List[float]:
        """Extract numeric values from a range specification"""
        values = []
        
        # Handle single cell reference
        if re.match(r'^[A-Z]+\d+$', range_str):
            value = self._extract_cell_value(data, range_str)
            if value is not None:
                values.append(value)
        
        # Handle range reference (A1:B10)
        elif ':' in range_str:
            start, end = range_str.split(':')
            # Extract all values in range
            # This is a simplified implementation
            values = self._extract_range_values(data, start, end)
        
        return [v for v in values if v is not None and not pd.isna(v)]
    
    def _extract_cell_value(self, data: pd.DataFrame, cell_ref: str) -> Any:
        """Extract value from a cell reference"""
        # Parse cell reference (e.g., A1 -> row 0, col 0)
        col_match = re.match(r'([A-Z]+)', cell_ref)
        row_match = re.search(r'(\d+)', cell_ref)
        
        if col_match and row_match:
            col = col_match.group(1)
            row = int(row_match.group(1)) - 1  # Convert to 0-based index
            
            # Convert column letter to index
            col_idx = self._column_to_index(col)
            
            if row < len(data) and col_idx < len(data.columns):
                return data.iloc[row, col_idx]
        
        return None
    
    def _column_to_index(self, col: str) -> int:
        """Convert Excel column letter to index"""
        result = 0
        for char in col:
            result = result * 26 + (ord(char) - ord('A') + 1)
        return result - 1
    
    def _extract_range_values(self, data: pd.DataFrame, start: str, end: str) -> List[float]:
        """Extract values from a range (simplified implementation)"""
        # This is a simplified implementation
        # In a full implementation, you'd parse the range properly
        values = []
        try:
            # For now, just extract some values from the data
            for i in range(min(10, len(data))):
                for j in range(min(5, len(data.columns))):
                    value = data.iloc[i, j]
                    if isinstance(value, (int, float)) and not pd.isna(value):
                        values.append(value)
        except:
            pass
        return values
    
    def _create_condition_mask(self, data: pd.DataFrame, condition: Dict[str, Any]) -> pd.Series:
        """Create a boolean mask based on a condition"""
        if condition['type'] == 'cell_comparison':
            left_value = self._extract_cell_value(data, condition['left_operand'])
            right_value = condition['right_operand']
            
            # Try to convert right_value to numeric if left_value is numeric
            if isinstance(left_value, (int, float)):
                try:
                    right_value = float(right_value)
                except:
                    pass
            
            operator = condition['operator']
            if operator == '>':
                return pd.Series(left_value > right_value, index=data.index)
            elif operator == '<':
                return pd.Series(left_value < right_value, index=data.index)
            elif operator == '>=':
                return pd.Series(left_value >= right_value, index=data.index)
            elif operator == '<=':
                return pd.Series(left_value <= right_value, index=data.index)
            elif operator == '==':
                return pd.Series(left_value == right_value, index=data.index)
            elif operator == '!=':
                return pd.Series(left_value != right_value, index=data.index)
        
        # Default to False
        return pd.Series(False, index=data.index)
    
    def learn_filtering_operations(self, data: pd.DataFrame, filter_conditions: List[Dict[str, Any]]) -> List[FilterCondition]:
        """
        Learn filtering operations from data and conditions
        
        Args:
            data: DataFrame being filtered
            filter_conditions: List of filter conditions applied
            
        Returns:
            List of learned FilterCondition objects
        """
        learned_filters = []
        
        for condition in filter_conditions:
            filter_condition = FilterCondition(
                column=condition.get('column', ''),
                operator=condition.get('operator', '=='),
                value=condition.get('value'),
                logical_operator=condition.get('logical_operator', 'AND')
            )
            learned_filters.append(filter_condition)
        
        self.filter_patterns.extend(learned_filters)
        return learned_filters
    
    def apply_learned_filters(self, data: pd.DataFrame, filters: List[FilterCondition] = None) -> pd.DataFrame:
        """
        Apply learned filtering operations to data
        
        Args:
            data: DataFrame to filter
            filters: List of FilterCondition objects (uses learned filters if None)
            
        Returns:
            Filtered DataFrame
        """
        if filters is None:
            filters = self.filter_patterns
        
        filtered_data = data.copy()
        
        for filter_condition in filters:
            if filter_condition.column in filtered_data.columns:
                mask = self._apply_filter_condition(filtered_data[filter_condition.column], 
                                                  filter_condition.operator, 
                                                  filter_condition.value)
                filtered_data = filtered_data[mask]
        
        return filtered_data
    
    def _apply_filter_condition(self, series: pd.Series, operator: str, value: Any) -> pd.Series:
        """Apply a single filter condition to a series"""
        if operator == '>':
            return series > value
        elif operator == '<':
            return series < value
        elif operator == '>=':
            return series >= value
        elif operator == '<=':
            return series <= value
        elif operator == '==':
            return series == value
        elif operator == '!=':
            return series != value
        elif operator == 'contains':
            return series.astype(str).str.contains(str(value), na=False)
        elif operator == 'starts_with':
            return series.astype(str).str.startswith(str(value), na=False)
        elif operator == 'ends_with':
            return series.astype(str).str.endswith(str(value), na=False)
        else:
            return pd.Series(True, index=series.index)
    
    def get_learned_patterns(self) -> Dict[str, CalculationPattern]:
        """Get all learned calculation patterns"""
        return self.learned_patterns.copy()
    
    def get_filter_patterns(self) -> List[FilterCondition]:
        """Get all learned filter patterns"""
        return self.filter_patterns.copy()
    
    def save_learned_patterns(self, filepath: str):
        """Save learned patterns to file"""
        import json
        
        patterns_data = {
            'calculation_patterns': {
                pid: {
                    'pattern_id': pattern.pattern_id,
                    'description': pattern.description,
                    'formula_logic': {
                        'formula_type': pattern.formula_logic.formula_type.value,
                        'operation': pattern.formula_logic.operation,
                        'source_range': pattern.formula_logic.source_range,
                        'target_range': pattern.formula_logic.target_range,
                        'conditions': pattern.formula_logic.conditions,
                        'parameters': pattern.formula_logic.parameters,
                        'dependencies': pattern.formula_logic.dependencies,
                        'result_type': pattern.formula_logic.result_type,
                        'confidence': pattern.formula_logic.confidence
                    },
                    'examples': pattern.examples,
                    'frequency': pattern.frequency,
                    'success_rate': pattern.success_rate
                }
                for pid, pattern in self.learned_patterns.items()
            },
            'filter_patterns': [
                {
                    'column': f.column,
                    'operator': f.operator,
                    'value': f.value,
                    'logical_operator': f.logical_operator
                }
                for f in self.filter_patterns
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(patterns_data, f, indent=2)
    
    def load_learned_patterns(self, filepath: str):
        """Load learned patterns from file"""
        import json
        
        with open(filepath, 'r') as f:
            patterns_data = json.load(f)
        
        # Load calculation patterns
        for pid, pattern_data in patterns_data.get('calculation_patterns', {}).items():
            formula_logic_data = pattern_data['formula_logic']
            formula_logic = FormulaLogic(
                formula_type=FormulaType(formula_logic_data['formula_type']),
                operation=formula_logic_data['operation'],
                source_range=formula_logic_data.get('source_range'),
                target_range=formula_logic_data.get('target_range'),
                conditions=formula_logic_data.get('conditions', []),
                parameters=formula_logic_data.get('parameters', {}),
                dependencies=formula_logic_data.get('dependencies', []),
                result_type=formula_logic_data.get('result_type', 'unknown'),
                confidence=formula_logic_data.get('confidence', 1.0)
            )
            
            pattern = CalculationPattern(
                pattern_id=pattern_data['pattern_id'],
                description=pattern_data['description'],
                formula_logic=formula_logic,
                examples=pattern_data.get('examples', []),
                frequency=pattern_data.get('frequency', 1),
                success_rate=pattern_data.get('success_rate', 1.0)
            )
            
            self.learned_patterns[pid] = pattern
        
        # Load filter patterns
        for filter_data in patterns_data.get('filter_patterns', []):
            filter_condition = FilterCondition(
                column=filter_data['column'],
                operator=filter_data['operator'],
                value=filter_data['value'],
                logical_operator=filter_data.get('logical_operator', 'AND')
            )
            self.filter_patterns.append(filter_condition)
