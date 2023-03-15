import pandas as pd
import cv2

court_pos = {'baseline_top' :((286, 561), (1379, 561)), \
             'baseline_bottom':((286, 2935), (1379, 2935)),\
             'net':((286, 1748), (1379, 1748)), \
             'left_court_line':((286, 561), (286, 2935)),
             'right_court_line':((1379, 561), (1379, 2935)),\
             'left_inner_line' : ((423, 561), (423, 2935)),\
             'right_inner_line' : ((1242, 561), (1242, 2935)),\
             'middle_line' : ((832, 1110), (832, 2386)),\
             'top_inner_line' : ((423, 1110), (1242, 1110)),\
             'bottom_inner_line' : ((423, 2386), (1242, 2386)),\
             'top_extra_part' : (832.5, 580),\
             'bottom_extra_part' : (832.5, 2910)}

def find_court_corners(section, court_pos):
    
    net               = court_pos['net']
    top_inner_line    = court_pos['top_inner_line']
    bottom_inner_line = court_pos['bottom_inner_line']
    middle_line       = court_pos['middle_line']
    left_inner_line   = court_pos['left_inner_line']
    right_inner_line  = court_pos['right_inner_line']
    
    
    if section == 'LT':
        x_min, y_min = top_inner_line[0][0], top_inner_line[0][1], 
        x_max, y_max = middle_line[0][0], net[0][1]
        
    elif section == 'RT':
        x_min, y_min = middle_line[0][0], middle_line[0][1], 
        x_max, y_max = top_inner_line[1][0], net[0][1]
        
    elif section == 'BL':
        x_min, y_min = bottom_inner_line[0][0], net[0][1], 
        x_max, y_max = middle_line[1][0], middle_line[1][1]
        
    elif section == 'BR':
        x_min, y_min = middle_line[0][0], net[0][1], 
        x_max, y_max = bottom_inner_line[1][0], bottom_inner_line[1][1]
        
    elif section == 'T':
        x_min, y_min = left_inner_line[0][0], left_inner_line[0][1]
        x_max, y_max = right_inner_line[1][0], right_inner_line[1][1]
        
    elif section == 'B':
        x_min, y_min = left_inner_line[0][0], left_inner_line[0][1]
        x_max, y_max = right_inner_line[1][0], right_inner_line[1][1]
        
        
    return (x_min, y_min, x_max, y_max)
        
    
def is_service_correct(bounce_point, player_pos, player, court_pos, margin=50):
    '''
        P1 = Top player
        P2 = Bottom Player
    '''
    
    ball_x, ball_y     = bounce_point
    player_x, player_y = player_pos


    middle_line        = court_pos['middle_line']
    
    if player == 'P1':
        if player_x > middle_line[0][0]:
            (x_min, y_min, x_max, y_max) =  find_court_corners('LT', court_pos)
            
        else:
            
            (x_min, y_min, x_max, y_max) =  find_court_corners('RT', court_pos)
              
    else:
        
        if player_x > middle_line[0][0]:
            (x_min, y_min, x_max, y_max) =  find_court_corners('BL', court_pos)
            
        else:
            (x_min, y_min, x_max, y_max) =  find_court_corners('BR', court_pos)
            
    
    if (ball_x > (x_min - margin)) and (ball_x < (x_max+margin)) and (ball_y > (y_min - margin)) and (ball_y < (y_max + margin)):
        return 'In'
            
    else:
        return 'Out'
    
    
def is_hitting_correct(bounce_point, player, court_pos, margin=50):
    
    try:
        ball_x, ball_y     = eval(bounce_point)

    except:
        ball_x, ball_y     = bounce_point

    
    if player == 'P1':
        (x_min, y_min, x_max, y_max) = find_court_corners('T', court_pos)
        
    else:
        (x_min, y_min, x_max, y_max) = find_court_corners('B', court_pos)
        

    if (ball_x > (x_min - margin)) and (ball_x < (x_max+margin)) and (ball_y > (y_min - margin)) and (ball_y < (y_max - margin)):
        return 'In'
            
    else:
        return 'Out'
            
        
def find_outcome_of_shot(row, first_shot, hitting_player, all_actions, hit_flag, last_action, is_action_true, action_series, frame, ind, tennis_tracking, volleyed):
    

    if row['Stroke_by'] != 'None':
        action_series   = ""
        hitting_player  =  row
        action_series  += "Short"
        player          = 'T' if hitting_player['Stroke_by'] == 'P1' else 'O'
       
        action_series   = action_series + f", {player}"
        
        

        if first_shot==False:
            hitting_player['Stroke_Type'] == 'Service/Smash'
            first_shot = True

        if len(all_actions) > 0:
            last_player_short = all_actions[-1].split(',')[1]

            print("last_player_short", last_player_short, "player", player, "hit_flag", hit_flag)
            if ((hit_flag == True) and (last_player_short != player)):
                tennis_tracking.at[ind, "Ball_Bounce_Outcome"] = "volleyed"
                print(volleyed)
                volleyed.append(ind)
                tennis_tracking.at[ind, "Ball_Bounce_Pos"]     = (row['Ball_POS'][0], row['Ball_POS'][1])
                hit_flag        = False

            else:
                hit_flag        = True

        else:
            hit_flag        = True

        

          
    if eval(row['Ball_bounced']) == True:      
        hit_flag        = False
        
        ball_hitting = row['Ball_bounced']
        
        if hitting_player['Stroke_Type'] == 'Service/Smash':
            
            action_series = action_series + f", Serve"
            if hitting_player['Stroke_by'] == 'P1':
                is_action_true = is_service_correct(row['Ball_POS'], hitting_player['Player_Near_End_Pos'], hitting_player['Stroke_by'], court_pos)
                action_series  = action_series + f", {row['Player_Near_End_Pos']}"
                
            else:
                is_action_true = is_service_correct(row['Ball_POS'], hitting_player['Player_Far_End_Pos'], hitting_player['Stroke_by'], court_pos)
                action_series  = action_series + f", {row['Player_Far_End_Pos']}"
    
            last_action   = 'Serve'
        
            
            
        else:
            # print("row ball pos", type(row['Ball_POS']))
            is_action_true = is_hitting_correct(row['Ball_POS'], hitting_player['Stroke_by'], court_pos)
            shot_type      = "Return" if last_action == 'Serve' else "Shot"
            action_series  = action_series + f", {shot_type}" 
            
            if hitting_player['Stroke_by'] == 'P1':      
                action_series = action_series + f", {row['Player_Near_End_Pos']}"
    
            else:
                action_series = action_series + f", {row['Player_Far_End_Pos']}"
            
            last_action   = shot_type


        tennis_tracking.at[ind, "Ball_Bounce_Outcome"] = is_action_true
        tennis_tracking.at[ind, "Ball_Bounce_Pos"]     = (row['Ball_POS'][0], row['Ball_POS'][1])

        if (hitting_player['Frame'] != None) and len(tennis_tracking.at[hitting_player['Frame'], "Ball_predict_point"])==0:

            tennis_tracking.at[hitting_player['Frame'], "Ball_predict_point"] = (row['Ball_POS'][0], row['Ball_POS'][1])

        action_series = action_series + f", {is_action_true}"
        action_series = action_series + f", {row['Ball_POS'][0]}, {row['Ball_POS'][1]}"
        action_series = action_series + f", {round(row['Time'], 4)}"
        
        all_actions.append(action_series)

    if tennis_tracking.at[ind, "Ball_Bounce_Outcome"] != 'None':
        is_action_true = tennis_tracking.at[ind, "Ball_Bounce_Outcome"]

        
    frame[270:350, 25:250] = 255.0
    
    if hit_flag==False:
        color = (0, 255, 0) if is_action_true=='In' else (0, 0, 255)
        frame = cv2.putText(frame, f'Ball: {is_action_true}',
                        (30, 330),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    else:
        frame = cv2.putText(frame, f'',
                            (30, 330),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    frame = cv2.putText(frame, f"{last_action} by {hitting_player['Stroke_by']}",
                        (30, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


    return first_shot, hitting_player, all_actions, hit_flag, last_action, is_action_true, action_series, frame, volleyed