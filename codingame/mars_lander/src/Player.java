import java.io.FileInputStream;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

class Edge
{
    public Node node1;
    public Node node2;
    public double length;

    public Edge(Node node1_, Node node2_, double length_)
    {
        node1 = node1_;
        node2 = node2_;
        length = length_;
    }

    public Node get_opposite_end(Node node)
    {
        return node1.id == node.id ? node2 : node1;
    }
}

class DijkstraInfo
{
    public Node node;
    public DijkstraInfo previous;
    public boolean is_handled = false;
    public double distance = Double.POSITIVE_INFINITY;

    public DijkstraInfo(Node node, DijkstraInfo previous, double distance)
    {
        this.node = node;
        this.previous = previous;
        this.distance = distance;
    }

    public DijkstraInfo(DijkstraInfo other)
    {
        node = other.node;
        previous = other.previous;
        is_handled = other.is_handled;
        distance = other.distance;
    }
}

class Node
{
    public int id;
    Point coordinates = null;
    public ArrayList<Edge> edges = new ArrayList<>();
    public boolean is_land_point = false;

    static int last_id = 0;

    public Node(Point coordinates_)
    {
        id = last_id++;
        coordinates = coordinates_;
    }

    public static void add_edge(Node node1, Node node2, double length)
    {
        Edge new_edge = new Edge(node1, node2, length);
        node1.edges.add(new_edge);
        node2.edges.add(new_edge);
    }
}

class Point
{
    double x;
    double y;

    public Point(double x_, double y_)
    {
        x = x_;
        y = y_;
    }

    public Point(Point other)
    {
        x = other.x;
        y = other.y;
    }

    public Point move_point(double direction, double magnitude)
    {
        return new Point(x + magnitude * Math.cos(direction), y + magnitude * Math.sin(direction));
    }
}

class LineSegment
{
    Point first;
    Point second;
    double normal;

    public LineSegment(Point first, Point second)
    {
        this.first = first;
        this.second = second;
        double dy = second.y - first.y;
        double dx = second.x - first.x;
        if (dx == 0)
        {
            if (dy > 0)
            {
                normal = Math.PI;
            } else if (dy < 0)
            {
                normal = 0;
            } else
            {
                System.out.println("Points coincide");
            }
        } else
        {
            normal = Math.atan(dy / dx) + Math.PI / 2;
            if (dx < 0)
            {
                normal += Math.PI;
            }
        }
    }

    public Point intersect_lines(LineSegment other)
    {
        double x1 = first.x;
        double x2 = second.x;
        double x3 = other.first.x;
        double x4 = other.second.x;
        double y1 = first.y;
        double y2 = second.y;
        double y3 = other.first.y;
        double y4 = other.second.y;
        double denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
        if (Player.roughly_equal(denominator, 0, Player.comparison_precision))
        {
            return null;
        }
        double new_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator;
        double new_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator;
        return new Point(new_x, new_y);
    }

    public Point intersect_segments(LineSegment other)
    {
        Point line_intersection = intersect_lines(other);
        if (line_intersection != null)
        {
            double left_x = Double.min(first.x, second.x);
            double right_x = Double.max(first.x, second.x);
            double other_left_x = Double.min(other.first.x, other.second.x);
            double other_right_x = Double.max(other.first.x, other.second.x);
            double left_margin = Double.max(left_x, other_left_x);
            double right_margin = Double.min(right_x, other_right_x);
            if (Player.roughly_ge(line_intersection.x, left_margin, Player.comparison_precision) && Player.roughly_le(line_intersection.x, right_margin, Player.comparison_precision))
            {
                return line_intersection;
            }
        }
        return null;
    }

    public double direction()
    {
        double direction = normal - Math.PI / 2;
        if (direction < 0)
        {
            direction += Math.PI * 2;
        }
        return direction;
    }

    public double reverse_direction()
    {
        double reverse_direction = normal + Math.PI / 2;
        if (reverse_direction > Math.PI * 2)
        {
            reverse_direction -= Math.PI * 2;
        }
        return reverse_direction;
    }

    public Point get_left_point()
    {
        return first.x < second.x ? first : second;
    }

    public Point get_right_point()
    {
        return first.x > second.x ? first : second;
    }
}

class Player
{
    final double landscape_shift = 100;
    final double mountain_offset = 200;
    final int max_land_v_speed = 39;
    final int max_land_h_speed = 19;
    final int max_rotate = 15;
    int max_h_speed = 30;
    final int max_power = 4;
    final int speed_precision = 2;
    final static double comparison_precision = 0.0001;
    final int coordinates_precision = 100;
    final double gravity_acceleration = 3.711;
    final double braking_distance = 1000;
    final int descent_power = 1;
    final double trajectory_fail_distance = coordinates_precision * 2;
    Point position;
    int h_speed;
    int v_speed;
    int fuel;
    int angle;
    int power;
    int abs_h_speed;
    int abs_v_speed;
    int v_speed_excess;
    int h_speed_excess;

    public static boolean roughly_equal(double n1, double n2, double tolerance)
    {
        return Math.abs(n1 - n2) < tolerance;
    }

    public static boolean roughly_equal(Point p1, Point p2, double tolerance)
    {
        return roughly_equal(p1.x, p2.x, tolerance) && roughly_equal(p1.y, p2.y, tolerance);
    }

    public static boolean roughly_ge(double n1, double n2, double tolerance)
    {
        return roughly_equal(n1, n2, tolerance) || n1 > n2;
    }

    public static boolean roughly_le(double n1, double n2, double tolerance)
    {
        return roughly_equal(n1, n2, tolerance) || n1 < n2;
    }

    public boolean roughly_equal(Point p1, Point p2)
    {
        return Player.roughly_equal(p1, p2, comparison_precision);
    }

    public boolean is_flat(LineSegment segment)
    {
        return segment.first.y == segment.second.y && segment.normal < Math.PI;
    }

    public void add_land_points(List<Node> graph_nodes, LineSegment flat_ground)
    {
        Point left = flat_ground.get_left_point();
        Point right = flat_ground.get_right_point();
        List<Node> new_nodes = new ArrayList<>();
        graph_nodes.stream().filter(node -> node.coordinates.x >= left.x && node.coordinates.x <= right.x && node.coordinates.y > left.y).
                forEach(node -> {
                    Node land_node = new Node(new Point(node.coordinates.x, left.y));
                    land_node.is_land_point = true;
                    new_nodes.add(land_node);
                });
        graph_nodes.addAll(new_nodes.stream().collect(Collectors.toList()));
    }

    public List<Node> get_graph_nodes(List<LineSegment> surface)
    {
        List<Node> graph_nodes = new ArrayList<>();
//        graph_nodes.add(new Node(new Point(start)));
        graph_nodes.add(new Node(new Point(surface.get(0).first)));
        LineSegment flat_ground = null;
        for (LineSegment segment : surface)
        {
            if (is_flat(segment))
            {
                flat_ground = segment;
                graph_nodes.remove(graph_nodes.size() - 1);
                Node.last_id -= 1;
                Node left_land = new Node(new Point(segment.first.move_point(0, mountain_offset)));
                Node right_land = new Node(new Point(segment.second.move_point(Math.PI, mountain_offset)));
                left_land.is_land_point = true;
                right_land.is_land_point = true;
                graph_nodes.add(left_land);
                graph_nodes.add(right_land);
            } else
            {
                graph_nodes.add(new Node(new Point(segment.second)));
            }
        }
        add_land_points(graph_nodes, flat_ground);
        return graph_nodes;
    }

    public List<Point> get_surface_points(Scanner in)
    {
        List<Point> points = new ArrayList<>();
        int surface_points = in.nextInt(); // the number of points used to draw the surface of Mars.
        System.err.println(surface_points);
        for (int i = 0; i < surface_points; ++i)
        {
            int x = in.nextInt(); // X coordinate of a surface point. (0 to 6999)
            int y = in.nextInt();
            System.err.format("%d %d\n", x, y);
            points.add(new Point(x, y));
        }
        return points;
    }

    public List<LineSegment> get_surface(List<Point> surface_points)
    {
        List<LineSegment> surface = new ArrayList<>();
        for (int i = 0; i < surface_points.size() - 1; ++i)
        {
            surface.add(new LineSegment(surface_points.get(i), surface_points.get(i + 1)));
        }
        return surface;
    }

    public List<LineSegment> enlarge_surface(List<LineSegment> surface, double trajectory_error)
    {
        List<LineSegment> enlarged_surface = surface.stream().
                map(s -> new LineSegment(s.first.move_point(s.normal, trajectory_error), s.second.move_point(s.normal, trajectory_error))).
                collect(Collectors.toList());
        List<Point> new_surface_points = new ArrayList<>();
        new_surface_points.add(enlarged_surface.get(0).first);
        for (int i = 0; i < enlarged_surface.size() - 1; ++i)
        {
            Point next_intersection = enlarged_surface.get(i).intersect_lines(enlarged_surface.get(i + 1));
            if (next_intersection == null)
            {
                next_intersection = enlarged_surface.get(i).second;
            }
            new_surface_points.add(next_intersection);
        }
        new_surface_points.add(enlarged_surface.get(enlarged_surface.size() - 1).second);
        return get_surface(new_surface_points);
    }

    public void update_status(Scanner in)
    {
        position = new Point(in.nextInt(), in.nextInt());
        h_speed = in.nextInt(); // the horizontal speed (in m/s), can be negative.
        v_speed = in.nextInt(); // the vertical speed (in m/s), can be negative.
        fuel = in.nextInt(); // the quantity of remaining fuel in liters.
        angle = in.nextInt(); // the rotation angle in degrees (-90 to 90).
        power = in.nextInt(); // the thrust power (0 to 4).
        abs_h_speed = Math.abs(h_speed);
        abs_v_speed = Math.abs(v_speed);
        v_speed_excess = Integer.max(0, abs_v_speed - max_land_v_speed);
        h_speed_excess = Integer.max(0, abs_h_speed - max_land_h_speed);
    }

    public boolean can_see_each_other(Point point1, Point point2, List<LineSegment> obstacles)
    {
        LineSegment sight_line = new LineSegment(point1, point2);
        for (int i = 0; i < obstacles.size(); ++i)
        {
            LineSegment current = obstacles.get(i);
            Point intersection = sight_line.intersect_segments(current);
            if (intersection != null && !roughly_equal(intersection, sight_line.second))
            {
                double normal_difference = sight_line.normal - current.normal;
                if (normal_difference >= 0 && normal_difference <= Math.PI)
                {
                    continue;
                }
                LineSegment first;
                LineSegment second;
                if (roughly_equal(intersection, current.first))
                {
                    if (i == 0)
                    {
                        return false;
                    }
                    first = obstacles.get(i - 1);
                    second = current;
                }
                else if (roughly_equal(intersection, current.second))
                {
                    if (i == obstacles.size() - 1)
                    {
                        return false;
                    }
                    first = current;
                    second = obstacles.get(i + 1);
                }
                else
                {
                    return false;
                }

                if (second.direction() > first.reverse_direction())
                {
                    if (sight_line.direction() > first.reverse_direction() && sight_line.direction() < second.direction())
                    {
                        return false;
                    }
                }
                else
                {
                    if (sight_line.direction() > first.reverse_direction() && sight_line.direction() <= Math.PI * 2 ||
                            sight_line.direction() >= 0 && sight_line.direction() < second.direction())
                    {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    public double euclidean_distance(Point point1, Point point2)
    {
        return Math.sqrt(Math.pow(point1.x - point2.x, 2) + Math.pow(point1.y - point2.y, 2));
    }

    public void connect_new_node(Node new_node, List<Node> path_graph, List<LineSegment> surface)
    {
        path_graph.stream().filter(other -> can_see_each_other(new_node.coordinates, other.coordinates, surface)).
                forEach(other -> Node.add_edge(new_node, other, euclidean_distance(new_node.coordinates, other.coordinates)));
    }

    public void connect_nodes(List<Node> path_graph, List<LineSegment> surface)
    {
        for (int i = 0; i < path_graph.size() - 1; ++i)
        {
            for (int j = i + 1; j < path_graph.size(); ++j)
            {
                Node node1 = path_graph.get(i);
                Node node2 = path_graph.get(j);
                if (can_see_each_other(node1.coordinates, node2.coordinates, surface))
                {
                    Node.add_edge(node1, node2, euclidean_distance(node1.coordinates, node2.coordinates));
                }
            }
        }
    }

    public List<DijkstraInfo> calculate_paths(List<Node> graph)
    {
        List<DijkstraInfo> nodes_info = graph.stream().map(n -> new DijkstraInfo(n, null, Double.POSITIVE_INFINITY)).
                collect(Collectors.toList());
        DijkstraInfo start = nodes_info.get(nodes_info.size() - 1);
        start.distance = 0;
        PriorityQueue<DijkstraInfo> info_queue = new PriorityQueue<>((a, b) -> Double.compare(a.distance, b.distance));
        info_queue.add(new DijkstraInfo(start));
        while (!info_queue.isEmpty())
        {
            DijkstraInfo current_info = nodes_info.get(info_queue.poll().node.id);
            if (current_info.is_handled)
            {
                continue;
            }
            for (Edge edge : current_info.node.edges)
            {
                DijkstraInfo adjacent = nodes_info.get(edge.get_opposite_end(current_info.node).id);
                if (current_info.distance + edge.length < adjacent.distance)
                {
                    adjacent.distance = current_info.distance + edge.length;
                    adjacent.previous = current_info;
                    info_queue.add(new DijkstraInfo(adjacent));
                }
            }
            current_info.is_handled = true;
        }
        return nodes_info;
    }

    public List<Point> restore_path(DijkstraInfo exit)
    {
        List<Point> result = new ArrayList<>();
        DijkstraInfo current = exit;
        while (current != null)
        {
            result.add(current.node.coordinates);
            current = current.previous;
        }
        Collections.reverse(result);
        return result;
    }

    public void add_node(List<Node> graph, List<LineSegment> surface, Point point)
    {
        Node new_node = new Node(new Point(point));
        connect_new_node(new_node, graph, surface);
        graph.add(new_node);
    }

    public List<Point> get_waypoints(List<Node> graph, List<LineSegment> surface, Point start)
    {
        add_node(graph, surface, start);
        List<DijkstraInfo> nodes_info = calculate_paths(graph);
        DijkstraInfo nearest_exit = nodes_info.stream().filter(n -> n.node.is_land_point).
                min((n1, n2) -> Double.compare(n1.distance, n2.distance)).get();
        return restore_path(nearest_exit);
    }

    public boolean point_reached(Point destination, Point current)
    {
        double distance = euclidean_distance(destination, current);
        boolean result = distance < coordinates_precision;
        System.err.format("next point: %f %f\n", destination.x, destination.y);
        System.err.format("distance: %f\n", distance);
        if (result)
        {
            System.err.format("POINT REACHED\n");
        }
        return result;
    }

    public boolean time_to_slow_down(List<Point> waypoints, int next_point_index)
    {
        return next_point_index == waypoints.size() - 1 &&
                euclidean_distance(position, waypoints.get(next_point_index)) < braking_distance &&
                (abs_h_speed > max_land_h_speed || abs_v_speed > max_land_v_speed);
    }

    public void give_command(int angle, int power, Scanner in)
    {
        System.out.format("%d %d\n", angle, power);
        update_status(in);
    }

    public void finish_landing(Scanner in)
    {
        while (Math.abs(h_speed) > max_land_h_speed + Math.abs(angle / max_rotate) * 2)
        {
            give_command((int) (90 * Math.signum(h_speed)), max_power, in);
        }
        while (true)
        {
            int power = 3;
            if (abs_v_speed > max_land_v_speed)
            {
                power = max_power;
            }
            give_command(0, power, in);
        }
    }

    public double solve_equation(double c1, double c2, double c3)
    {
        double a = Math.pow(c1, 2) + Math.pow(c2, 2);
        double b = -2 * c2 * c3;
        double c = Math.pow(c3, 2) - Math.pow(c1, 2);
        double x1 = (-b + Math.sqrt(Math.pow(b, 2) - 4 * a * c)) / (2 * a);
        double x2 = (-b - Math.sqrt(Math.pow(b, 2) - 4 * a * c)) / (2 * a);
        double right_x = roughly_equal(c1 * Math.sqrt(1 - Math.pow(x1, 2)) + c2 * x1 - c3, 0, comparison_precision) ? x1 : x2;
        System.err.format("Coefficients: %f %f %f %f %f %f %f %f\n", c1, c2, c3, a, b, c, x1, x2);
        return Math.acos(right_x);
    }

    public List<Integer> required_speed_optimal_command(double required_h, double required_v)
    {
        System.err.format("Required: %f %f\n", required_h, required_v);
        double gain_h = required_h - h_speed;
        if (Math.round(Math.abs(gain_h)) < speed_precision)
        {
            gain_h = 0;
        }
        double gain_v = required_v - v_speed;
        if (Math.round(Math.abs(gain_v)) < speed_precision)
        {
            gain_v = 0;
        }
        System.err.format("Remains: %f %f\n", gain_h, gain_v);
        if (gain_h == 0 && gain_v == 0)
        {
            return Arrays.asList(0, 4);
        }
        if (gain_h == 0)
        {
            int power = gain_v > 0 ? max_power : descent_power;
            return Arrays.asList(0, power);
        }
        else
        {
            double c1 = gain_v / Math.abs(gain_h);
            double c2 = -1;
            double rad_angle = 0;
            Stream<Integer> power_stream = IntStream.rangeClosed(1, max_power).boxed();
            if (required_v > v_speed)
            {
                power_stream = power_stream.sorted(Collections.reverseOrder());
            }
            List<Integer> powers = power_stream.collect(Collectors.toList());
            int power = 0;
            for (int next_power: powers)
            {
                power = next_power;
                double c3 = -gravity_acceleration / next_power;
                rad_angle = solve_equation(c1, c2, c3);
                if (rad_angle <= Math.PI / 2)
                {
                    break;
                }
            }
            System.err.format("power: %d, angle: %f\n", power, rad_angle);
            int desired_angle = (int) (Math.round(rad_angle * 180 / Math.PI) * -Math.signum(gain_h));
            return Arrays.asList(desired_angle, power);
        }
    }

    public void follow_course(List<Point> waypoints, Scanner in)
    {
        for (int i = 1; i < waypoints.size(); ++i)
        {
            Point next_point = waypoints.get(i);
            Point previous_point = waypoints.get(i - 1);
            double min_distance = euclidean_distance(position, next_point);
            double dx = next_point.x - previous_point.x;
            while (!point_reached(next_point, position))
            {
                double distance = euclidean_distance(position, next_point);
                min_distance = Double.min(min_distance, distance);
                if (distance - min_distance > trajectory_fail_distance)
                {
                    System.err.format("Min distance: %f\n", min_distance);
                    System.err.format("Trajectory failed, recalculating...\n");
                    return;
                }
                if (time_to_slow_down(waypoints, i))
                {
                    max_h_speed = max_land_h_speed;
                }

                if (dx == 0)
                {
                    int power = 3;
                    if (abs_v_speed > max_land_v_speed)
                    {
                        power = max_power;
                    }
                    else if (abs_v_speed < max_land_v_speed - gravity_acceleration)
                    {
                        power = 0;
                    }
                    give_command(0, power, in);
                    continue;
                }
                double new_dx = next_point.x - position.x;
                double new_dy = next_point.y - position.y;
                double new_ratio = Math.abs(new_dy) / new_dx;
                double required_h = max_h_speed * Math.signum(new_dx);
                double required_v = Math.abs(required_h * new_ratio) * Math.signum(new_dy);
                if (Math.abs(required_v) > max_land_v_speed)
                {
                    required_v = max_land_v_speed * Math.signum(new_dy);
                    required_h = Math.abs(required_v / new_ratio) * Math.signum(new_dx);
                }
                List<Integer> optimal_command = required_speed_optimal_command(required_h, required_v);
                give_command(optimal_command.get(0), optimal_command.get(1), in);
            }
        }
        finish_landing(in);
    }

    public void report_trajectory(List<Point> waypoints)
    {
        System.err.format("Waypoints:\n");
        for (Point point: waypoints)
        {
            System.err.format("%f %f\n", point.x, point.y);
        }
    }

    public void land()
    {
        Scanner in = new Scanner(System.in);
        List<LineSegment> surface = get_surface(get_surface_points(in));
        List<LineSegment> enlarged_surface = enlarge_surface(surface, landscape_shift);
        update_status(in);
        List<Node> path_graph = get_graph_nodes(enlarged_surface);
        connect_nodes(path_graph, enlarged_surface);
        while (true)
        {
            List<Point> waypoints = get_waypoints(path_graph, enlarged_surface, position);
            report_trajectory(waypoints);
            follow_course(waypoints, in);
        }
    }

    public static void main(String args[])
    {
        try
        {
            System.setIn(new FileInputStream("input"));
        }
        catch (Exception e)
        {
        }

        Player player = new Player();
        player.land();
    }
}