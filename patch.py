import os

with open('d:/trading/signal_generator.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add tracking vars
content = content.replace(
    '    candidates: list[dict] = []\n    regime_confidence = market_profile["score"]',
    '    candidates: list[dict] = []\n    generated_candidates_count = 0\n    accepted_signals_list = []\n    rejected_signals_list = []\n    regime_confidence = market_profile["score"]'
)

# 2. Add count
content = content.replace(
    '        for direction in ("CALL", "PUT"):\n            session_str, session_detail',
    '        for direction in ("CALL", "PUT"):\n            generated_candidates_count += 1\n            session_str, session_detail'
)

# 3. Inject rejected tracking
content = content.replace(
    '                logger.debug(\n                    "[Slot] Skip %s %s — vol_zone=%s", ist_time_str, direction, vol_zone\n                )\n                continue',
    '                logger.debug(\n                    "[Slot] Skip %s %s — vol_zone=%s", ist_time_str, direction, vol_zone\n                )\n                rejected_signals_list.append((ist_time_str, direction, f"vol_zone={vol_zone}"))\n                continue'
)

content = content.replace(
    '                    ist_time_str, direction, slot_atr_ratio, atr_ratio_min, atr_ratio_max,\n                )\n                continue',
    '                    ist_time_str, direction, slot_atr_ratio, atr_ratio_min, atr_ratio_max,\n                )\n                rejected_signals_list.append((ist_time_str, direction, f"atr_ratio={slot_atr_ratio:.2f} out of bounds [{atr_ratio_min:.2f},{atr_ratio_max:.2f}]"))\n                continue'
)

content = content.replace(
    '                    _regime_min_score(regime),\n                )\n                continue',
    '                    _regime_min_score(regime),\n                )\n                rejected_signals_list.append((ist_time_str, direction, f"prob_score={prob_result.probability_score:.1f} < regime_floor={_regime_min_score(regime):.0f}"))\n                continue'
)

content = content.replace(
    '                    prob_result.probability_score, prob_result.signal_tier,\n                )\n                continue',
    '                    prob_result.probability_score, prob_result.signal_tier,\n                )\n                rejected_signals_list.append((ist_time_str, direction, f"reversal-heavy requires STRONG (score={prob_result.probability_score:.1f})"))\n                continue'
)

content = content.replace(
    '                    agreement.agreement_score, agreement.total_voters, agreement.tier,\n                )\n                continue',
    '                    agreement.agreement_score, agreement.total_voters, agreement.tier,\n                )\n                rejected_signals_list.append((ist_time_str, direction, f"agreement={agreement.agreement_score}/{agreement.total_voters} ({agreement.tier})"))\n                continue'
)

content = content.replace(
    '                    "[Learn] Veto %s %s — learning adj=%d",\n                    ist_time_str, direction, adj_legacy,\n                )\n                continue',
    '                    "[Learn] Veto %s %s — learning adj=%d",\n                    ist_time_str, direction, adj_legacy,\n                )\n                rejected_signals_list.append((ist_time_str, direction, f"learning veto (adj={adj_legacy})"))\n                continue'
)

content = content.replace(
    '                    ist_time_str, direction, learned_strength, blended_conf, fallback_threshold,\n                )\n                continue',
    '                    ist_time_str, direction, learned_strength, blended_conf, fallback_threshold,\n                )\n                rejected_signals_list.append((ist_time_str, direction, f"DB strength={learned_strength} & blended_conf={blended_conf:.1f} < {fallback_threshold:.1f}"))\n                continue'
)

content = content.replace(
    '                    "[Pattern] Reject %s %s: %s", ist_time_str, direction, live_reason\n                )\n                continue',
    '                    "[Pattern] Reject %s %s: %s", ist_time_str, direction, live_reason\n                )\n                rejected_signals_list.append((ist_time_str, direction, f"live confirmation failed: {live_reason}"))\n                continue'
)

content = content.replace(
    '            if adjusted_conf < fallback_threshold:\n                continue',
    '            if adjusted_conf < fallback_threshold:\n                rejected_signals_list.append((ist_time_str, direction, f"adjusted_conf={adjusted_conf:.1f} < {fallback_threshold:.1f}"))\n                continue'
)

# 4. Add accepted signals
content = content.replace(
    '            candidates.append({',
    '            accepted_signals_list.append((ist_time_str, direction))\n\n            candidates.append({'
)

# 5. Add final log outputs before returning final
end_block = '''
    logger.info(f"\\n--- Signal Generation Summary ({today_ist}) ---")
    logger.info(f"Generated candidates: {generated_candidates_count}")
    logger.info(f"Accepted signals: {len(accepted_signals_list)}")
    logger.info(f"Rejected signals: {len(rejected_signals_list)}")
    for r in rejected_signals_list:
        logger.debug(f"Rejected {r[0]} {r[1]} -> {r[2]}")
    logger.info(f"Final signal count: {len(final)}\\n")

    return final
'''

content = content.replace(
    '    return final',
    end_block
)

with open('d:/trading/signal_generator.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Success')
