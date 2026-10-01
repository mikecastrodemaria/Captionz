# Bench Captionz — Giant_monster_destroys_the_city.png

Date: 2026-09-02. Image resized from 1648×2944 to 1024 px (203 KB JPEG). Prompt: "Paragraph (natural caption)", temperature 0.2, keep_alive=0.

**Total** = wall time for a cold call. **Load time** = loading into VRAM (paid once per batch in the app thanks to keep_alive). **Per image** = Total − Load time, the actual cost in a batch.

| Model | Accuracy | Per image | Total | Load time | Generation | tok/s | Words |
|---|:---:|---:|---:|---:|---:|---:|---:|
| acc100/muse-glimmer-heretic:latest | 5/5 | **7.0s** | 20.2s | 13.2s | 6.2s | 72.0 | 168 |
| tinyrick/gemma-4-31B-it-uncensored-heretic-vision-llmfan46:Q4_K_M | 5/5 | **14.9s** | 24.2s | 9.3s | 14.1s | 58.6 | 124 |
| aratan/Qwable-agent-9B-Claude-Fable-5-heretic-GGUF:Q6_K | 4.5/5 | **6.2s** | 12.3s | 6.1s | 5.7s | 143.2 | 147 |
| qwen3.6:latest | 4.5/5 | **6.5s** | 17.3s | 10.8s | 5.9s | 210.1 | 143 |
| satgeze/qwenpaw-9b-heretic-1m:latest | 4.5/5 | **8.5s** | 16.1s | 7.6s | 8.0s | 135.8 | 167 |
| nutboy02/Agents-A1-4B-Kimi-heretic:latest | 4/5 | **4.3s** | 8.7s | 4.4s | 3.9s | 233.0 | 141 |
| Fermi/Cydonia-24B-v4.3-heretic-vision:Q4_K_M | 3.5/5 | **3.1s** | 10.7s | 7.6s | 2.6s | 92.5 | 191 |
| qwen3-vl:8b | 3.5/5 | **4.6s** | 9.6s | 5.0s | 4.1s | 194.9 | 170 |
| iliafed/nemotron3-quant:latest | 3/5 | **2.1s** | 10.8s | 8.7s | 1.6s | 300.7 | 194 |

## Quality notes

- **muse-glimmer-heretic:latest** (5/5): Most complete: Empire State Building, water towers, plane, smoke, running child, and the monster's foot. No errors.
- **gemma-4-31B-it-uncensored-heretic-vision-llmfan46:Q4_K_M** (5/5): Everything is present, no hallucinations, and the most concise text. Slowest generation.
- **Qwable-agent-9B-Claude-Fable-5-heretic-GGUF:Q6_K** (4.5/5): Complete (Empire State Building, plane, smoke, person, falling object). Good quality-to-speed ratio.
- **qwen3.6:latest** (4.5/5): Complete, precise vocabulary (kaiju, hatching, stippling); same inaccuracy of "rooftops" instead of clouds.
- **qwenpaw-9b-heretic-1m:latest** (4.5/5): Complete and accurate; says the person "runs on rooftops" although they run on clouds.
- **Agents-A1-4B-Kimi-heretic:latest** (4/5): Very good for a 4B model: Empire State Building, plane, water towers, and person. Misses the smoke. Fastest.
- **Cydonia-24B-v4.3-heretic-vision:Q4_K_M** (3.5/5): Style and mood are well captured; includes the person and water towers but misses the plane and smoke, and invents fire escapes.
- **qwen3-vl:8b** (3.5/5): Very detailed, but places the smoke "on the monster's head" and adds a "burning building": two factual errors.
- **nemotron3-quant:latest** (3/5): Says smoke is "coming from the monster's side" and the person is "in the streets": both are wrong. Very verbose, but extremely fast generation.

## Generated captions

### acc100/muse-glimmer-heretic:latest

A black-and-white ink illustration in a loose, cross-hatched cartoon style depicts a towering, hairy monster with two small horns, wide white eyes and a jagged toothy grin looming over a dense cityscape that evokes New York, with the Empire State Building visible on the left and rows of detailed apartment blocks and water towers in the foreground. The giant creature dominates the upper center of the composition, its long thin arms reaching down toward fluffy cloud-like masses that obscure the skyline, while a small child in dark clothing runs across the clouds below, one of the monster’s legs appearing to chase them. A small airplane flies near the monster’s head and a plume of smoke or debris rises from the left side of the frame. The drawing uses high-contrast black ink on an off-white paper ground with minimal shading, giving a sketchy, expressive linework feel. The overall mood is whimsical and surreal with a playful, slightly mischievous tension between the enormous friendly-looking monster and the tiny fleeing figure.

### tinyrick/gemma-4-31B-it-uncensored-heretic-vision-llmfan46:Q4_K_M

A towering, black, fuzzy monster with a wide, toothy grin and large, staring eyes dominates a vertical composition, looming over a dense cityscape. The creature's body is rendered with intricate, vertical ink hatching, contrasting against a cream-colored background. Below the monster, a small human figure is depicted running across a layer of stylized, fluffy clouds that float above a forest of skyscrapers and urban buildings. To the right, a small airplane flies near the monster's head, while a plume of smoke or dust drifts in the upper left background. The artwork is a monochrome pen-and-ink illustration characterized by a hand-drawn, sketchy style with high contrast and flat lighting. The overall mood is surreal and whimsical, evoking a sense of imaginative scale and playful chaos.

### aratan/Qwable-agent-9B-Claude-Fable-5-heretic-GGUF:Q6_K

A towering, monstrous creature dominates the center of this vertical ink illustration, its rounded body densely cross-hatched in black against a cream background. The beast features two small horns, wide circular eyes, and a gaping maw filled with sharp triangular teeth, looming menacingly over a cityscape that evokes New York City with recognizable skyscrapers like the Empire State Building visible on the left. Below the monster, a layer of fluffy clouds separates it from the street level, where a tiny figure runs in panic to the right while another object tumbles through the air on the left. An airplane flies near the creature's head, and smoke billows dramatically from the upper left corner, adding to the chaotic atmosphere. The style is reminiscent of classic comic book art or children's storybook illustrations, utilizing high-contrast black ink lines and detailed textures to create a whimsical yet slightly threatening mood.

### qwen3.6:latest

A towering, ink-drawn kaiju dominates the vertical composition, its massive, textured body filled with dense vertical hatching rising above a thick layer of clouds that obscures its lower legs. The creature features large, round eyes, pointed ears, and a wide, toothy grin, exuding a chaotic yet slightly cartoonish energy as it looms over a cityscape reminiscent of New York, identifiable by the Empire State Building silhouette on the left where a trail of debris or smoke falls from the sky. Below the monster's feet, a panicked figure runs across detailed rooftops in the foreground, while an airplane flies perilously close to the beast's head on the right. The scene is rendered in stark black and white, utilizing heavy linework and stippling to create depth and texture, evoking a classic comic book or woodcut aesthetic that balances apocalyptic destruction with a whimsical, illustrative mood.

### satgeze/qwenpaw-9b-heretic-1m:latest

In a striking black-and-white illustration, a gigantic, furry monster with jagged horns and a massive, tooth-filled mouth dominates the upper half of the frame, emerging from a sea of clouds above a city skyline. The monster's body is rendered with intricate cross-hatching, giving it a textured, almost shaggy appearance, while its wide, circular eyes and outstretched arms convey a sense of awe-inspiring presence. Below, the cityscape is filled with detailed skyscrapers and water towers, including a prominent spire reminiscent of the Empire State Building, all partially obscured by the cloud layer. A small, silhouetted figure runs across the rooftops in the lower center, providing a sense of scale and adding a touch of frantic energy to the scene. An airplane flies near the monster's head, and a dark, indistinct object falls from the sky on the left, near a plume of smoke. The artwork utilizes a high-contrast, sketchy style with loose lines and a cream-colored background, evoking a mood that is simultaneously surreal, chaotic, and slightly humorous.

### nutboy02/Agents-A1-4B-Kimi-heretic:latest

A towering, dark-furred monster with sharp teeth and horns looms over a sprawling cityscape in this vertical black-and-white illustration, creating a sense of surreal scale. The creature, textured with dense cross-hatching, dominates the upper half of the frame, its mouth open in a roar while a plane flies past its head against the pale sky. Below, a detailed urban environment filled with buildings and water towers stretches across the bottom third, where a small figure runs away in panic towards the right, emphasizing the monster's immense size. To the left, the spire of the Empire State Building anchors the scene, suggesting a specific metropolitan setting. The style is reminiscent of hand-drawn comic book art, utilizing stark contrasts and expressive line work to create a whimsical yet tense mood, capturing a moment of chaotic intrusion where the giant beast disrupts everyday life.

### Fermi/Cydonia-24B-v4.3-heretic-vision:Q4_K_M

A massive, monstrous creature with a toothy grin dominates the sky above a sprawling cityscape, its dark, textured form stretching across the upper portion of the composition. The beast appears to be emerging from or descending toward the clouds that blanket the middle ground, its large eyes fixed downward as if surveying the urban environment below. A small human figure can be seen running across the clouds in the middle distance, creating a sense of scale that emphasizes the creature's enormous size. The city below consists of densely packed buildings rendered in a lighter tone, with recognizable architectural elements including skyscrapers, water towers, and fire escapes. The illustration employs a high-contrast black and white palette with cross-hatching used extensively to create texture and depth, particularly on the creature's body. The style is reminiscent of woodcut prints or vintage illustrations, featuring bold lines and simplified forms that give the scene a timeless, slightly unsettling quality. The overall mood is one of surreal menace mixed with whimsy, as the fantastical creature looms over the mundane city below, suggesting themes of impending danger, the collision of worlds, or perhaps a child's nightmare made manifest.

### qwen3-vl:8b

A towering, cartoonish monster with a striped body, large circular eyes, sharp jagged teeth, and smoke billowing from its head dominates a surreal cityscape rendered in stark black-and-white line art. The setting features a dense urban skyline with recognizable skyscrapers (including a spire-topped tower), water towers, and buildings partially obscured by swirling clouds, while a small human figure runs frantically through the clouds below the monster, a commercial airplane flies near the monster’s head, and a building emits smoke and flames in the upper left. The composition centers the monster as the focal point, with dynamic elements like the running figure and aircraft creating a sense of urgency. Lighting is implied through heavy cross-hatching on the monster’s body and building details, contrasting with the lighter clouds and sky. The monochromatic color scheme—strictly black ink on off-white paper—enhances the sketchy, hand-drawn style, which blends whimsical cartooning with surreal, slightly ominous tension. The overall mood is chaotic and urgent, balancing playful absurdity with underlying danger from the monstrous presence and city chaos.

### iliafed/nemotron3-quant:latest

A towering, cat‑like monster dominates the upper half of the composition, its massive, textured body rendered in dense cross‑hatching that gives it a looming, almost tactile presence as it rises above a sprawling cityscape; the creature’s wide, toothy grin and bulging eyes convey a menacing yet cartoonish menace, while a plume of smoke and debris erupts from its left side, suggesting recent destruction. Below, a dense cluster of skyscrapers and lower‑rise buildings fills the foreground, their outlines crisp against a sky dotted with fluffy clouds, a small airplane soaring in the upper right, and a lone figure sprinting frantically through the streets, adding a sense of scale and urgency. The scene is rendered entirely in black and white, employing stark contrast and varied line weights to create depth, with the monster’s dark silhouette set against lighter clouds and sky, and the city’s detailed architecture providing a busy, chaotic backdrop. The overall mood is frantic and ominous, evoking a surreal, comic‑book‑style nightmare where the colossal beast threatens the urban environment, while the dynamic composition—centered on the monster, with the fleeing figure and distant plane guiding the eye—heightens the sense of impending danger and frantic escape.
