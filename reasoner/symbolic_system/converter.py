import os
import json
import time
import random
import warnings
import argparse
from itertools import permutations, product, combinations

from extended_definition import ExtendedDefinition
from logic_parser import LogicParser
from logic_solver import LogicSolver
from utils import isNumber, hasNumber, isAlgebra, findAlgebra, sort_points, sort_angle

from kanren import Relation, facts
from kanren import run, var, conde

import sympy

def _same(list1, list2):
    return any([pair[0] == pair[1] for pair in product(list1, list2)])

def sympy2latex(x):
    if isinstance(x, sympy.Basic):
        latex = sympy.latex(x)
        return latex
    else:   
        return x

def Text2Logic(text, debug_mode=False):
    parser = LogicParser(ExtendedDefinition(debug=debug_mode))

    # Define diagram primitive elements
    parser.logic.point_positions = text['point_positions']

    isLetter = lambda ch: ch.upper() and len(ch) == 1
    parser.logic.define_point([_ for _ in parser.logic.point_positions if isLetter(_)])
    if debug_mode:
        print(parser.logic.point_positions)

    lines = text['line_instances']  # ['AB', 'AC', 'AD', 'BC', 'BD', 'CD']
    for line in lines:
        line = line.strip()
        if len(line) == 2 and isLetter(line[0]) and isLetter(line[1]):
            parser.logic.define_line(line[0], line[1])

    circles = text['circle_instances']  # ['O']
    for point in circles:
        parser.logic.define_circle(point)

    # Parse logic forms
    logic_forms = text['logic_forms']
    def sort_func(x):
        if "Find" in x:
            return 3
        if "AltitudeOf" in x or "HeightOf" in x:
            return 2
        if "Perpendicular" in x:
            return 1
        return -1
        
    logic_forms = sorted(logic_forms, key=sort_func)

    target = None
    for logic_form in logic_forms:
        if logic_form.strip() != "":
            if debug_mode:
                print("The logic form is", logic_form)
            
            if logic_form.find('Find') != -1:
                target = parser.findTarget(parser.parse(logic_form)) # ['Value', 'A', 'C']
            else:
                parse_tree = parser.parse(logic_form) # ['Equals', ['LengthOf', ['Line', 'A', 'C']], '10']
                parser.dfsParseTree(parse_tree)
    
    return parser, target

def Logic2Text(logic, reserved_info=None, debug_mode=False):
    ret_text = {}
    if reserved_info != None:
        ret_text['point_positions'] = reserved_info['point_positions']
        ret_text['line_instances'] = reserved_info['line_instances']
        ret_text['circle_instances'] = reserved_info['circle_instances']

    logic_forms = []

    # Circle(O)
    circle_list = logic.find_all_circles()
    logic_forms.extend(["Circle({})".format(circle) for circle in circle_list])
    # Triangle(A,B,C)
    triangle_list = logic.find_all_triangles()
    logic_forms.extend(["Triangle({},{},{})".format(triangle[0], triangle[1], triangle[2]) for triangle in triangle_list])
    # Quadrilateral(A,B,C,D)
    quadrilateral_list = logic.find_all_quadrilaterals()
    logic_forms.extend(["Quadrilateral({},{},{},{})".format(quadrilateral[0], quadrilateral[1], quadrilateral[2], quadrilateral[3]) for quadrilateral in quadrilateral_list])
    # Pentagon(A,B,C,D,E)
    pentagon_list = logic.find_all_pentagons()
    logic_forms.extend(["Pentagon({},{},{},{},{})".format(pentagon[0], pentagon[1], pentagon[2], pentagon[3], pentagon[4]) for pentagon in pentagon_list])

    # PointLiesOnCircle(A, Circle(O))  Ignored radius
    pointOnCircle_list = logic.find_all_points_on_circles()
    for pointOnCircle in pointOnCircle_list:
        logic_forms.extend(["PointLiesOnCircle({},Circle({}))".format(point, pointOnCircle[1]) for point in pointOnCircle[0]])

    # Equals(LengthOf(Line(A, B)), Value) or Equals(LengthOf(Line(A, B)), LengthOf(Line(C, D)))
    length_list = logic.find_all_irredundant_lines_with_length()
    for length in length_list:
        if f"line_{length[0]}{length[1]}" == str(length[2]) or f"line_{length[1]}{length[0]}" == str(length[2]):
            continue
        else:
            logic_forms.extend(["Equals(line_{}{},{})".format(length[0], length[1], length[2])])
    # Equals(MeasureOf(Angle(A, B, C)), Value) or Equals(MeasureOf(Angle(A, B, C)), MeasureOf(Angle(D, E, F)))
    angleMeasure_list = logic.find_all_irredundant_angle_measures()
    for angleMeasure in angleMeasure_list:
        if f"angle_{angleMeasure[0]}{angleMeasure[1]}{angleMeasure[2]}" == str(angleMeasure[3]) or f"angle_{angleMeasure[2]}{angleMeasure[1]}{angleMeasure[0]}" == str(angleMeasure[3]):
            continue
        else:
            logic_forms.extend(["Equals(angle_{}{}{},{})".format(angleMeasure[0], angleMeasure[1], angleMeasure[2], angleMeasure[3])])
    # Equals(MeasureOf(Arc(O, A, B)), Value) or Equals(MeasureOf(Arc(O, A, B)), Arc(O, C, D))
    arcMeasure_list = logic.fine_all_arc_measures()
    for arcMeasure in arcMeasure_list:
        if f"arc_{arcMeasure[0]}{arcMeasure[1]}{arcMeasure[2]}" == str(arcMeasure[3]):
            continue
        else:
            logic_forms.extend(["Equals(arc_{}{}{},{})".format(arcMeasure[0], arcMeasure[1], arcMeasure[2], arcMeasure[3])])
    
    # Parallel(Line(A, B), Line(C, D))
    parallel_list = logic.find_all_parallels()
    logic_forms.extend(["Parallel(Line({},{}),Line({},{}))".format(parallel[0][0], parallel[0][1], parallel[1][0], parallel[1][1]) for parallel in parallel_list])

    # Equals(m, n)
    x = var()
    y = var()
    res = run(0, (x, y), logic.Equal(x, y))
    logic_forms.extend(["Equals({},{})".format(sym1, sym2) for sym1, sym2 in res])
    res = run(0, (x, y), logic.Equation(x, y))
    logic_forms.extend(["Equals({},{})".format(sym1, sym2) for sym1, sym2 in res])
    
    # Similar(Triangle(A,B,C),Triangle(D,E,F))
    a = var()
    b = var()
    c = var()
    d = var()
    e = var()
    f = var()
    # Congruent(Triangle(A,B,C),Triangle(D,E,F))
    congruentTriangle_list = run(0, ((a,b,c),(d,e,f)), logic.CongruentTriangle(a,b,c,d,e,f))
    logic_forms.extend(["Congruent(Triangle({}),Triangle({}))".format(','.join(congruentTriangle[0]), ','.join(congruentTriangle[1])) for congruentTriangle in congruentTriangle_list])
    similarTriangle_list = run(0, ((a,b,c),(d,e,f)), logic.SimilarTriangle(a,b,c,d,e,f))
    logic_forms.extend(["Similar(Triangle({}),Triangle({}))".format(','.join(similarTriangle[0]), ','.join(similarTriangle[1])) for similarTriangle in similarTriangle_list])

    # Similar(Polygon(),Polygon())
    polygons = logic.find_all_similar_polygons()
    for poly1, poly2 in polygons:
        logic_forms.extend(["Similar(Polygon({}),Polygon({}))".format(','.join(poly1), ','.join(poly2))])


    ret_text['logic_forms'] = logic_forms
    
    return ret_text

def getTargetObject(logic, target):
    if target[0] == 'Value':
        if len(target) == 5:
            return 'arc_'+''.join(target[2:])
        if len(target) == 4:
            return 'angle_'+''.join(sort_angle(target[1:]))
        if len(target) == 3:
            return 'line_'+''.join(sorted(target[1:]))
        if len(target) == 2:
            return 'variable_'+target[1]
    if target[0] == 'Area':
        if len(target) == 2:
            return 'circle_'+target[1]
        if len(target) == 4:
            return 'triangle_'+''.join(sorted(target[1:]))
        if len(target) == 5:
            return 'polygon_'+''.join(sort_points(target[1:]))
    if target[0] == 'Perimeter':
        if len(target) == 2:
            circle = target[1]
            return 'line_'+''.join(sorted(logic.find_points_on_circle(circle)[0]+circle))
        else:
            poly = target[1:]
            return ['line_'+''.join(sorted([poly[i], poly[(i+1)%len(poly)]])) for i in range(len(poly))]
    if target[0] == 'Sector':
        O,A,B = target[1:]
        return ['angle_'+sort_angle(A+O+B), 'line_'+''.join(sorted(O+A))]
    if target[0] in ["SinOf", "CosOf", "TanOf", "CotOf", "HalfOf", "SquareOf", "SqrtOf"]:
        return getTargetObject(logic, target[1])
    if target[0] in ["RatioOf", "Add", "Mul", "SumOf"]:
        return [getTargetObject(logic, target[i]) for i in range(1, len(target))]
    if target[0] == 'ScaleFactorOf':
        if target[1][0] == "Shape" and len(target[1]) == 2:    
            line = (target[1][1], target[2][1])
            points = logic.find_all_points_on_line(line)
            O = (set(points) - set(line)).pop()
            return ['line_'+''.join(sorted(O+line[1])), 'line_'+''.join(O+line[0])]
        else:
            shape1 = target[1] if type(target[1][1]) == str else target[1][1]
            shape2 = target[2] if type(target[2][1]) == str else target[2][1]
            return [getTargetObject(['Area', *shape1[1:]]), getTargetObject(['Area', *shape2[1:]])]



def Logic2Graph(logic, target):
    # Node: Point, Line, Angle, Arc, Circle, Triangle, Polygon
    # Relation:
    # <Point, Point>: Connected
    # <Point, Line>: EndPoint, LiesOnLine
    # <Point, Angle>: Vertex, SidePoint
    # <Point, Arc>: Center, EndPoint
    # <Point, Circle>: Center, LiesOnCircle
    # <Point, Triangle> / <Point, Polygon>: Vertex
    # <Line, Line>: Equal, Parallel, Perpendicular
    # <Line, Triangle> / <Line, Polygon>: Side
    # <Angle, Angle>: Equal
    # <Angle, Triangle> / <Angle, Polygon>: Interior
    # <Arc, Arc>: Equal
    # <Triangle, Triangle>: Congruent
    # <Triangle, Triangle>: Similar
    # <Polygon, Polygon>: Similar
    node = []
    node_type = []
    node_attr = []
    target_node = []
    edge_st_index = []
    edge_ed_index = []
    edge_attr = []

    points = sorted(logic.find_all_points())
    
    lines = sorted(logic.find_all_irredundant_lines())
    length = []
    for line in lines:
        val = logic.find_line_with_length(line)
        v = findAlgebra(val)
        if v != None:
            if isNumber(v):
                v = float(v)
            length.append(v)
        else:
            length.append('None')

    angles = []
    angleMeasures = []
    all_angles = sorted(logic.find_all_irredundant_angles())
    for angle in all_angles:
        val = logic.find_angle_measure(angle)
        v = findAlgebra(val)
        if v != None:
            if isNumber(v):
                v = float(v)
            if v != 180:
                angles.append(angle)
                angleMeasures.append(v)

    arcs = []
    arcMeasures = []
    all_arcs = sorted(logic.find_all_arcs())
    for arc in all_arcs:
        val = logic.find_arc_measure(arc)
        v = findAlgebra(val)
        if v != None:
            if isNumber(v):
                v = float(v)
            if v != 180:
                arcs.append(arc)
                arcMeasures.append(v)
    
    circles = logic.find_all_circles()

    triangles = sorted(logic.find_all_triangles())
    tri_lines = []
    tri_angles = []
    triangleAreas = []
    for tri in triangles:
        tri_lines.extend([sorted([tri[i],tri[(i+1)%3]]) for i in range(3)])
        tri_angles.extend([(tri[0], tri[2], tri[1]), (tri[0], tri[1], tri[2]), (tri[1], tri[0], tri[2])])
        AreaSymbol = sympy.Symbol("AreaOf(Polygon({}))".format(','.join(sort_points(tri))) )
        if AreaSymbol in logic.variables and isAlgebra(logic.variables[AreaSymbol]):
            v = logic.variables[AreaSymbol]
            if isNumber(v):
                v = float(v)
            triangleAreas.append(v)
        else:
            triangleAreas.append('None')   
    Vertex = []
    Vertex_R = []
    Interior = []
    Interior_R = []
    Side = []
    Side_R = []
    for i, tri in enumerate(triangles):
        for j in range(3):
            Vertex.append((tri[j], 'triangle_'+''.join(tri)))
            Vertex_R.append(('triangle_'+''.join(tri), tri[j]))
            t_line = tri_lines[i*3+j]
            if t_line not in lines:
                lines.append(t_line)
                length.append('None')
            Side.append(('line_'+''.join(t_line), 'triangle_'+''.join(tri)))
            Side_R.append(('triangle_'+''.join(tri), 'line_'+''.join(t_line)))

            t_angle = logic.get_same_angle_key(tri_angles[i*3+j])
            t_measure = logic.find_angle_measure(t_angle)[0] if hasNumber(logic.find_angle_measure(t_angle)) else 'None'
            NOT_IN = True
            for angle in angles:
                if logic.check_same_angle(t_angle, angle):
                    t_angle = angle
                    t_measure = logic.find_angle_measure(angle)
                    NOT_IN = False
                    break
            if NOT_IN: 
                if isNumber(t_measure):
                    t_measure = float(t_measure)
                if not isAlgebra(t_measure):
                    t_measure = 'None'
                angles.append(t_angle)
                angleMeasures.append(t_measure)
            Interior.append(('angle_'+''.join(t_angle), 'triangle_'+''.join(tri)))
            Interior_R.append(('triangle_'+''.join(tri), 'angle_'+''.join(t_angle)))

    polygons = sorted(logic.find_all_quadrilaterals() + logic.find_all_pentagons())
    poly_lines = []
    poly_angles = []
    polygonAreas = []
    for poly in polygons:
        poly_lines.append([sorted([poly[i], poly[(i+1)%len(poly)]]) for i in range(len(poly))])
        poly_angles.append([[poly[i], poly[(i+1)%len(poly)], poly[(i+2)%len(poly)]] for i in range(len(poly))])
        AreaSymbol = sympy.Symbol("AreaOf(Polygon({}))".format(','.join(sort_points(poly))) )
        if AreaSymbol in logic.variables and isAlgebra(logic.variables[AreaSymbol]):
            t_measure = logic.variables[AreaSymbol]
            if isNumber(t_measure):
                t_measure = float(t_measure)
            polygonAreas.append(t_measure)
        else:
            polygonAreas.append('None')

    for i, poly in enumerate(polygons):
        for j in range(len(poly)):
            Vertex.append((poly[j], 'polygon_'+''.join(poly)))
            Vertex_R.append(('polygon_'+''.join(poly), poly[j]))
            t_line = poly_lines[i][j]
            if t_line not in lines:
                lines.append(t_line)
                length.append('None')
            Side.append(('line_'+''.join(t_line), 'polygon_'+''.join(poly)))
            Side_R.append(('polygon_'+''.join(poly), 'line_'+''.join(t_line)))

            t_angle = logic.get_same_angle_key(poly_angles[i][j])
            t_measure = logic.find_angle_measure(t_angle)[0] if hasNumber(logic.find_angle_measure(t_angle)) else 'None'
            NOT_IN = True
            for angle in angles:
                if logic.check_same_angle(t_angle, angle):
                    t_angle = angle
                    t_measure = logic.find_angle_measure(angle)
                    NOT_IN = False
                    break
            if NOT_IN: 
                if isNumber(t_measure):
                    t_measure = float(t_measure)
                if not isAlgebra(t_measure):
                    t_measure = 'None'
                angles.append(t_angle)
                angleMeasures.append(t_measure)
            Interior.append(('angle_'+''.join(t_angle), 'polygon_'+''.join(poly)))
            Interior.append(('polygon_'+''.join(poly), 'angle_'+''.join(t_angle)))

    # <Point, Point>
    connected_points = []
    for line in lines:
        connected_points.append((line[0], line[1]))
        connected_points.append((line[1], line[0]))
    # <Point, Line>
    endpoint_line = []
    endpoint_R_line = []
    for line in lines:
        endpoint_line.append((line[0], 'line_'+''.join(line)))
        endpoint_line.append((line[1], 'line_'+''.join(line)))
        endpoint_R_line.append(('line_'+''.join(line), line[0]))
        endpoint_R_line.append(('line_'+''.join(line), line[1]))
    pointLiesOnLine = []
    pointLiesOnLine_R = []
    p, a, b = var(), var(), var()
    res = run(0, (p,a,b), logic.PointLiesOnLine(p, a, b))
    for p,a,b in list(res):
        if a > b: a,b = b,a
        pointLiesOnLine.append((p, f'line_{a}{b}'))
        pointLiesOnLine_R.append((f'line_{a}{b}', p))
    # <Point, Angle>
    vertex_angle = []
    vertex_R_angle = []
    sidePoint_angle = []
    sidePoint_R_angle = []
    for angle in angles:
        vertex_angle.append((angle[1], 'angle_'+''.join(angle)))
        vertex_R_angle.append(('angle_'+''.join(angle), angle[1]))
        sidePoint_angle.append((angle[0], 'angle_'+''.join(angle)))
        sidePoint_angle.append((angle[2], 'angle_'+''.join(angle)))
        sidePoint_R_angle.append(('angle_'+''.join(angle), angle[0]))
        sidePoint_R_angle.append(('angle_'+''.join(angle), angle[0]))
    # <Point, Arc>
    center_arc = []
    center_R_arc = []
    endpoint_arc = []
    endpoint_R_arc = []
    for arc in arcs:
        center_arc.append((arc[0], 'arc_'+''.join(arc)))
        center_R_arc.append(('arc_'+''.join(arc), arc[0]))
        endpoint_arc.append((arc[1], 'arc_'+''.join(arc)))
        endpoint_arc.append((arc[2], 'arc_'+''.join(arc)))
        endpoint_R_arc.append(('arc_'+''.join(arc), arc[1]))
        endpoint_R_arc.append(('arc_'+''.join(arc), arc[2]))
    # <Point, Circle>
    center_cirlce = []
    center_R_cirlce = []
    pointLiesOnCircle = []
    pointLiesOnCircle_R = []
    for circle in circles:
        center_cirlce.append((circle, 'circle_'+circle))
        center_R_cirlce.append(('circle_'+circle, circle))
        for point in logic.find_points_on_circle(circle):
            pointLiesOnCircle.append((point, 'circle_'+circle))
            pointLiesOnCircle_R.append(('circle_'+circle, point))
    # <Line, Line>
    Equal = []
    Parallel = []
    Perpendicular = []
    for s in logic.EqualLineSet:
        for l1, l2 in permutations(s, 2):
            Equal.append(('line_'+''.join(l1), 'line_'+''.join(l2)))
    for l1, l2 in logic.find_all_parallels():
        l1 = ''.join(sorted(l1))
        l2 = ''.join(sorted(l2))
        Parallel.append(('line_'+''.join(l1), 'line_'+''.join(l2)))
        Parallel.append(('line_'+''.join(l2), 'line_'+''.join(l1)))
    for l1, l2 in logic.find_all_perpendicule():
        l1 = ''.join(sorted(l1))
        l2 = ''.join(sorted(l2))
        Perpendicular.append(('line_'+''.join(l1), 'line_'+''.join(l2)))
        Perpendicular.append(('line_'+''.join(l2), 'line_'+''.join(l1)))
    # <Angle, Angle>
    for s in logic.EqualAngleSet:
        for a1, a2 in permutations(s, 2):
            a1 = logic.get_same_angle_key(a1)
            a2 = logic.get_same_angle_key(a2)
            Equal.append(('angle_'+''.join(a1), 'angle_'+''.join(a2)))
    # <Arc, Arc>
    for s in logic.EqualArcSet:
        for a1, a2 in permutations(s, 2):
            Equal.append(('arc_'+''.join(a1), 'arc_'+''.join(a2)))
    # <Triangle, Triangle>
    # A small fault: side match
    Congruent = []
    for tri1, tri2 in logic.find_all_congruent_triangles():
        Congruent.append(('triangle_'+''.join(sorted(tri1)), 'triangle_'+''.join(sorted(tri2))))
        Congruent.append(('triangle_'+''.join(sorted(tri2)), 'triangle_'+''.join(sorted(tri1))))
    Similar = []
    for tri1, tri2 in logic.find_all_similar_triangles():
        Similar.append(('triangle_'+''.join(sorted(tri1)), 'triangle_'+''.join(sorted(tri2))))
        Similar.append(('triangle_'+''.join(sorted(tri2)), 'triangle_'+''.join(sorted(tri1))))
    for poly1, poly2 in logic.find_all_similar_polygons():
        Similar.append(('polygon_'+''.join(sort_points(poly1)), 'polygon_'+''.join(sort_points(poly2))))
        Similar.append(('polygon_'+''.join(sort_points(poly2)), 'polygon_'+''.join(sort_points(poly1))))
    
    node.extend([point for point in points])
    node_type.extend(['Point' for point in points])
    node_attr.extend(['None' for point in points])
    node.extend(['line_'+''.join(line) for line in lines])
    node_type.extend(['Line' for line in lines])
    node_attr.extend([l for l in length])
    node.extend(['angle_'+''.join(angle) for angle in angles])
    node_type.extend(['Angle' for angle in angles])
    node_attr.extend([angleMeasure for angleMeasure in angleMeasures])
    node.extend(['arc_'+''.join(arc) for arc in arcs])
    node_type.extend(['Arc' for arc in arcs])
    node_attr.extend([arcMeasure for arcMeasure in arcMeasures])
    node.extend(['circle_'+circle for circle in circles])
    node_type.extend(['Circle' for circle in circles])
    node_attr.extend(['None' for circle in circles])
    node.extend(['triangle_'+''.join(tri) for tri in triangles])
    node_type.extend(['Triangle' for tri in triangles])
    node_attr.extend([a for a in triangleAreas])
    node.extend(['polygon_'+''.join(poly) for poly in polygons])
    node_type.extend(['Polygon' for poly in polygons])
    node_attr.extend([a for a in polygonAreas])

    targetObj = getTargetObject(logic, target)
    if type(targetObj) != list: targetObj = [targetObj]
    for t in targetObj:
        if t.startswith('variable_'):
            variable = t.split('_')[-1]
            for i, attr in enumerate(node_attr):
                if isinstance(attr, sympy.Basic) and sympy.Symbol(variable) in attr.free_symbols:
                    target_node.append(node[i])
        else:
            if t in node:
                target_node.append(t)
    
    for connected_st, connected_ed in connected_points:
        edge_st_index.append(node.index(connected_st))
        edge_ed_index.append(node.index(connected_ed))
        edge_attr.append('Connected')
    for endpoint, line in endpoint_line:
        edge_st_index.append(node.index(endpoint))
        edge_ed_index.append(node.index(line))
        edge_attr.append('Endpoint')
    for line, endpoint in endpoint_R_line:
        edge_st_index.append(node.index(line))
        edge_ed_index.append(node.index(endpoint))
        edge_attr.append('Endpoint_R')
    for point, line in pointLiesOnLine:
        edge_st_index.append(node.index(point))
        edge_ed_index.append(node.index(line))
        edge_attr.append('LiesOnLine')
    for line, point in pointLiesOnLine_R:
        edge_st_index.append(node.index(line))
        edge_ed_index.append(node.index(point))
        edge_attr.append('LiesOnLine_R')
    for point, angle in vertex_angle:
        edge_st_index.append(node.index(point))
        edge_ed_index.append(node.index(angle))
        edge_attr.append('Vertex')
    for angle, point in vertex_R_angle:
        edge_st_index.append(node.index(angle))
        edge_ed_index.append(node.index(point))
        edge_attr.append('Vertex_R')
    for point, angle in sidePoint_angle:
        edge_st_index.append(node.index(point))
        edge_ed_index.append(node.index(angle))
        edge_attr.append('Sidepoint')
    for angle, point in sidePoint_R_angle:
        edge_st_index.append(node.index(angle))
        edge_ed_index.append(node.index(point))
        edge_attr.append('Sidepoint_R')
    for point, arc in center_arc:
        edge_st_index.append(node.index(point))
        edge_ed_index.append(node.index(arc))
        edge_attr.append('Center')
    for arc, point in center_R_arc:
        edge_st_index.append(node.index(arc))
        edge_ed_index.append(node.index(point))
        edge_attr.append('Center_R')
    for point, arc in endpoint_arc:
        edge_st_index.append(node.index(point))
        edge_ed_index.append(node.index(arc))
        edge_attr.append('Endpoint')
    for arc, point in endpoint_R_arc:
        edge_st_index.append(node.index(arc))
        edge_ed_index.append(node.index(point))
        edge_attr.append('Endpoint_R')
    for point, circle in center_cirlce:
        edge_st_index.append(node.index(point))
        edge_ed_index.append(node.index(circle))
        edge_attr.append('Center')
    for circle, point in center_R_cirlce:
        edge_st_index.append(node.index(circle))
        edge_ed_index.append(node.index(point))
        edge_attr.append('Center_R')
    for point, circle in pointLiesOnCircle:
        edge_st_index.append(node.index(point))
        edge_ed_index.append(node.index(circle))
        edge_attr.append('LiesOnCircle')
    for circle, point in pointLiesOnCircle_R:
        edge_st_index.append(node.index(circle))
        edge_ed_index.append(node.index(point))
        edge_attr.append('LiesOnCircle_R')
    for point, poly in Vertex:
        edge_st_index.append(node.index(point))
        edge_ed_index.append(node.index(poly))
        edge_attr.append('Vertex')
    for poly, point in Vertex_R:
        edge_st_index.append(node.index(poly))
        edge_ed_index.append(node.index(point))
        edge_attr.append('Vertex_R')
    for st, ed in Equal:
        edge_st_index.append(node.index(st))
        edge_ed_index.append(node.index(ed))
        edge_attr.append('Equal')
    for l1, l2 in Parallel:
        edge_st_index.append(node.index(l1))
        edge_ed_index.append(node.index(l2))
        edge_attr.append('Parallel')
    for l1, l2 in Perpendicular:
        edge_st_index.append(node.index(l1))
        edge_ed_index.append(node.index(l2))
        edge_attr.append('Perpendicular')
    for line, poly in Side:
        edge_st_index.append(node.index(line))
        edge_ed_index.append(node.index(poly))
        edge_attr.append('Side')
    for poly, line in Side_R:
        edge_st_index.append(node.index(poly))
        edge_ed_index.append(node.index(line))
        edge_attr.append('Side_R')
    for angle, poly in Interior:
        edge_st_index.append(node.index(angle))
        edge_ed_index.append(node.index(poly))
        edge_attr.append('Interior')
    for poly, angle in Interior_R:
        edge_st_index.append(node.index(poly))
        edge_ed_index.append(node.index(angle))
        edge_attr.append('Interior_R')
    for poly1, poly2 in Congruent:
        edge_st_index.append(node.index(poly1))
        edge_ed_index.append(node.index(poly2))
        edge_attr.append('Congruent')
    for poly1, poly2 in Similar:
        edge_st_index.append(node.index(poly1))
        edge_ed_index.append(node.index(poly2))
        edge_attr.append('Similar')

    edge_index = [edge_st_index, edge_ed_index]
    node_attr = [str(_).rstrip("0").rstrip(".")  if isNumber(_) else str(_) for _ in node_attr]

    return {"node": node,
            "node_type": node_type,
            "node_attr": node_attr,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "target_node": target_node}


if __name__ == "__main__":
    diagram_logic_forms_json_file = "../data/geometry3k/logic_forms/diagram_logic_forms_annot.json"
    text_logic_forms_json_file = "../data/geometry3k/logic_forms/text_logic_forms_annot_dissolved.json"
    diagram_logic_forms_json = json.load(open(diagram_logic_forms_json_file, 'r'))
    text_logic_forms_json = json.load(open(text_logic_forms_json_file, 'r'))
    for q_id in range(2401, 3002):
        q_id = 0
        text = diagram_logic_forms_json[str(q_id)]
        text["logic_forms"] = text.pop("diagram_logic_forms")
        text["logic_forms"].extend(text_logic_forms_json[str(q_id)]["text_logic_forms"])
        parser, target = Text2Logic(text, True)
        solver = LogicSolver(parser.logic, target)
        solver.initSearch()
        graph = Logic2Graph(solver.logic, target)
        
    
       

        
    