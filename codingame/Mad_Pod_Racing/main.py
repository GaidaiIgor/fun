def main():
    boosts = 1

    while True:
        x, y, next_checkpoint_x, next_checkpoint_y, next_checkpoint_dist, next_checkpoint_angle = [int(i) for i in input().split()]
        opponent_x, opponent_y = [int(i) for i in input().split()]

        if next_checkpoint_angle > 90 or next_checkpoint_angle < -90:
            thrust = 0
        else:
            thrust = 100
        if next_checkpoint_angle == 0 and next_checkpoint_dist > 5000 and boosts > 0:
            thrust = 'BOOST'
            boosts -= 1

        print(f'{next_checkpoint_x} {next_checkpoint_y} {thrust}')


main()
