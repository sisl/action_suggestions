function POMDPs.reward(pomdp::TagPOMDP, s::TagState, a::Int)
    if isterminal(pomdp, s)
        return 0.0
    end
    if a == ACTIONS_DICT[:tag]
        if s.r_pos == s.t_pos
            return pomdp.tag_reward
        end
        return pomdp.tag_penalty
    end
    return pomdp.step_penalty
end
