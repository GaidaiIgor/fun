import java.io.FileInputStream;
import java.util.*;
import java.util.stream.Collectors;

enum Action
{
    SPEED, JUMP, UP, DOWN, WAIT, SLOW
}

class Node
{
    public Node parent = null;
    public List<Node> children = null;
    Action action = null;
    int x = -1;
    List<Integer> ys = new ArrayList<>();
    int speed = 0;
    int tree_size = 0;

    public Node(Action action_, int x_, List<Integer> ys_, int speed_, int tree_size_, Node parent_)
    {
        action = action_;
        x = x_;
        ys = ys_;
        speed = speed_;
        tree_size = tree_size_;
        parent = parent_;
    }
}

class Player
{
    int total_bikes;
    int minimum_bikes;
    String[] road = new String[4];
    int depth = 4;
    boolean path_found = false;
    Stack<String> exit_commands = new Stack<>();

    public void check_exit(List<Node> nodes)
    {
        for (Node node: nodes)
        {
            if (node != null && node.x >= road[0].length() && !path_found)
            {
                while (node.parent != null)
                {
                    exit_commands.push(node.action.toString());
                    node = node.parent;
                }
                path_found = true;
                break;
            }
        }
    }

    public Node create_speed_node(Node parent)
    {
        Node new_node = new Node(Action.SPEED, parent.x + parent.speed + 1, new ArrayList<>(parent.ys), parent.speed + 1, 0, parent);
//        if (new_node.x >= road[0].length())
//        {
//            return new_node;
//        }
        for (int i = 0; i < new_node.ys.size(); ++i)
        {
            int y = new_node.ys.get(i);
            for (int x = parent.x + 1; x <= Integer.min(new_node.x, road[0].length() - 1); ++x)
            {
                if (road[y].charAt(x) == '0')
                {
                    new_node.ys.remove(i);
                    i -= 1;
                    break;
                }
            }
        }
        return new_node.ys.size() < minimum_bikes ? null : new_node;
    }

    public Node create_jump_node(Node parent)
    {
        Node new_node = new Node(Action.JUMP, parent.x + parent.speed, new ArrayList<>(parent.ys), parent.speed, 0, parent);
        if (new_node.x >= road[0].length())
        {
            return new_node;
        }
        for (int i = 0; i < new_node.ys.size(); ++i)
        {
            int y = new_node.ys.get(i);
            if (road[y].charAt(new_node.x) == '0')
            {
                new_node.ys.remove(i);
                i -= 1;
            }
        }
        return new_node.ys.size() < minimum_bikes ? null : new_node;
    }

    public Node create_up_node(Node parent)
    {
        if (parent.ys.get(0) == 0)
        {
            return create_wait_node(parent);
        }
        Node new_node = new Node(Action.UP, parent.x + parent.speed,
                new ArrayList<>(parent.ys).stream().map(i -> i - 1).collect(Collectors.toList()),
                parent.speed, 0, parent);
        for (int i = 0; i < new_node.ys.size(); ++i)
        {
            int y = new_node.ys.get(i);
            boolean is_removed = false;
            for (int x = parent.x + 1; x <= Integer.min(new_node.x, road[0].length() - 1); ++x)
            {
                if (road[y].charAt(x) == '0')
                {
                    new_node.ys.remove(i);
                    i -= 1;
                    is_removed = true;
                    break;
                }
            }
            if (is_removed)
            {
                continue;
            }
            for (int x = parent.x + 1; x < Integer.min(new_node.x, road[0].length()); ++x)
            {
                if (road[y + 1].charAt(x) == '0')
                {
                    new_node.ys.remove(i);
                    i -= 1;
                    break;
                }
            }
        }
        return new_node.ys.size() < minimum_bikes ? null : new_node;
    }

    public Node create_down_node(Node parent)
    {
        if (parent.ys.get(parent.ys.size() - 1) == 3)
        {
            return create_wait_node(parent);
        }
        Node new_node = new Node(Action.DOWN, parent.x + parent.speed,
                new ArrayList<>(parent.ys).stream().map(i -> i + 1).collect(Collectors.toList()),
                parent.speed, 0, parent);
        if (new_node.x >= road[0].length())
        {
            return new_node;
        }
        for (int i = 0; i < new_node.ys.size(); ++i)
        {
            int y = new_node.ys.get(i);
            boolean is_removed = false;
            for (int x = parent.x + 1; x <= Integer.min(new_node.x, road[0].length() - 1); ++x)
            {
                if (road[y].charAt(x) == '0')
                {
                    new_node.ys.remove(i);
                    i -= 1;
                    is_removed = true;
                    break;
                }
            }
            if (is_removed)
            {
                continue;
            }
            for (int x = parent.x + 1; x < Integer.min(new_node.x, road[0].length()); ++x)
            {
                if (road[y - 1].charAt(x) == '0')
                {
                    new_node.ys.remove(i);
                    i -= 1;
                    break;
                }
            }
        }
        return new_node.ys.size() < minimum_bikes ? null : new_node;
    }

    public Node create_wait_node(Node parent)
    {
        Node new_node = new Node(Action.WAIT, parent.x + parent.speed, new ArrayList<>(parent.ys), parent.speed, 0, parent);
        if (new_node.x >= road[0].length())
        {
            return new_node;
        }
        for (int i = 0; i < new_node.ys.size(); ++i)
        {
            int y = new_node.ys.get(i);
            for (int x = parent.x + 1; x <= Integer.min(new_node.x, road[0].length() - 1); ++x)
            {
                if (road[y].charAt(x) == '0')
                {
                    new_node.ys.remove(i);
                    i -= 1;
                    break;
                }
            }
        }
        return new_node.ys.size() < minimum_bikes ? null : new_node;
    }

    public Node create_slow_node(Node parent)
    {
        Node new_node = new Node(Action.SLOW, parent.x + parent.speed - 1, new ArrayList<>(parent.ys), parent.speed - 1, 0, parent);
        if (new_node.x >= road[0].length())
        {
            return new_node;
        }
        if (new_node.speed <= 0)
        {
            return null;
        }
        for (int i = 0; i < new_node.ys.size(); ++i)
        {
            int y = new_node.ys.get(i);
            for (int x = parent.x + 1; x <= Integer.min(new_node.x, road[0].length() - 1); ++x)
            {
                if (road[y].charAt(x) == '0')
                {
                    new_node.ys.remove(i);
                    i -= 1;
                    break;
                }
            }
        }
        return new_node.ys.size() < minimum_bikes ? null : new_node;
    }

    public void update_tree_size(Node node)
    {
        node.tree_size = 0;
        for (Node child: node.children)
        {
            node.tree_size += child.tree_size;
        }
        if (node.action == Action.SLOW)
        {
            update_tree_size(node.parent);
        }
    }

    public List<Node> get_next_layer(List<Node> current_layer)
    {
        List<Node> new_layer = new ArrayList<>();
        for (Node node: current_layer)
        {
            Node speed_node = create_speed_node(node);
            Node jump_node = create_jump_node(node);
            Node up_node = create_up_node(node);
            Node down_node = create_down_node(node);
            Node wait_node = create_wait_node(node);
            Node slow_node = create_slow_node(node);
            node.children = new ArrayList<>();
            Arrays.asList(speed_node, jump_node, up_node, down_node, wait_node, slow_node).stream().filter(child -> child != null).
                    forEach(child -> {
                        node.children.add(child);
                        new_layer.add(child);
                    });
            check_exit(node.children);

            node.tree_size += node.children.size();
            if (node.parent != null && node.parent.children.get(node.parent.children.size() - 1) == node)
            {
                update_tree_size(node.parent);
            }
        }
        return new_layer;
    }

    public void collect_children(Node root, List<Node> children)
    {
        if (root.children == null)
        {
            children.add(root);
            return;
        }
        root.children.stream().forEach(next_child -> collect_children(next_child, children));
    }

    public Node decide(Node root)
    {
        List<Integer> tree_sizes = root.children.stream().map(c -> c != null ? c.tree_size : 0).collect(Collectors.toList());
        int best_index = tree_sizes.indexOf(tree_sizes.stream().filter(e -> e > 0).findFirst().get());
        Node new_root = root.children.get(best_index);
        new_root.parent = null;
        System.out.println(new_root.action.toString());
        return new_root;
    }

    public void skip_input(Scanner in)
    {
        int speed = in.nextInt();
        System.err.println(speed);
        for (int i = 0; i < total_bikes; ++i)
        {
            int x = in.nextInt();
            int y = in.nextInt();
            int a = in.nextInt();
            System.err.format("%d %d %d\n", x, y, a);
        }
    }

    public void play()
    {
        Scanner in = new Scanner(System.in);
        total_bikes = in.nextInt(); // the amount of motorbikes to control
        System.err.println(total_bikes);
        minimum_bikes = in.nextInt(); // the minimum amount of motorbikes that must survive
        System.err.println(minimum_bikes);
        for (int i = 0; i < 4; ++i)
        {
            road[i] = in.next();
            System.err.println(road[i]);
        }

        int speed = in.nextInt(); // the motorbikes' SPEED
        System.err.println(speed);
        List<Integer> ys = new ArrayList<>();
        for (int i = 0; i < total_bikes; ++i)
        {
            in.nextInt(); // x coordinate of the motorbike
            int y = in.nextInt();
            ys.add(y);
            in.nextInt(); // indicates whether the motorbike is activated "1" or destroyed "0"
            System.err.format("0 %d 0\n", y);
        }

        List<Node> current_layer = new ArrayList<>();
        Node root = new Node(null, 0, ys, speed, 0, null);
        current_layer.add(root);
        for (int i = 0; i < depth; ++i)
        {
            current_layer = get_next_layer(current_layer);
        }

        // game loop
        while (true)
        {
            if (!path_found)
            {
                root = decide(root);
                current_layer = new ArrayList<>();
                collect_children(root, current_layer);
                get_next_layer(current_layer);
            }
            else
            {
                String next_command = exit_commands.empty() ? "WAIT" : exit_commands.pop();
                System.out.println(next_command);
            }
            skip_input(in);
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
        player.play();
    }
}