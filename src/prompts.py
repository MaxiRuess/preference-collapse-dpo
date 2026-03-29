"""Seed prompts and LLM-based augmentation.

Step 1 of the pipeline: start with hand-written seed prompts covering
AI-and-labor-economics questions where optimists and skeptics genuinely
disagree, then use Claude to augment to ~700 total.
"""

from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Seed prompts — decision-relevant questions about AI's economic impact
# Organized by category. Each prompt should elicit substantively different
# answers from optimist vs. skeptic personas.
# ---------------------------------------------------------------------------

SEED_PROMPTS: list[str] = [
    # --- Direct displacement: speed & scale ---
    "Will AI lead to mass unemployment in the next decade?",
    "Which jobs are most at risk of being fully automated by AI?",
    "Can AI replace knowledge workers like lawyers and accountants?",
    "How will AI affect entry-level jobs for recent graduates?",
    "Is the threat of AI job displacement overstated by the media?",
    "Will AI eliminate more white-collar or blue-collar jobs?",
    "How should a mid-career professional in finance think about AI displacement risk?",
    "Will AI make human creativity more or less economically valuable?",
    "How quickly will AI displace jobs compared to how quickly new jobs will emerge?",
    "Will AI displacement happen gradually enough for labor markets to adjust, or will there be sudden shocks?",
    "Is there a tipping point where AI capabilities cause rapid, widespread job loss?",
    "Will AI displace jobs faster in the next 10 years than outsourcing did in the previous 30?",
    "How many jobs worldwide will AI eliminate by 2040?",
    "Will AI displacement accelerate or decelerate as the technology matures?",
    "Is the current pace of AI adoption fast enough to cause labor market disruption within this decade?",

    # --- Direct displacement: cognitive vs. physical work ---
    "Can AI replicate human judgment in complex, ambiguous situations?",
    "Will AI be better at replacing routine cognitive work or non-routine cognitive work?",
    "How will AI affect jobs that require emotional intelligence and empathy?",
    "Will AI displace more jobs that involve decision-making or jobs that involve information processing?",
    "Can AI replace the diagnostic reasoning of experienced doctors?",
    "Will AI make human expertise in specialized fields more or less valuable?",
    "How will AI handle the tacit knowledge that experienced professionals rely on?",
    "Will AI struggle to replace jobs requiring physical dexterity and real-world navigation?",

    # --- Direct displacement: occupation-specific ---
    "Will AI replace most paralegal and legal research positions within a decade?",
    "How will AI affect the demand for human translators and interpreters?",
    "Will AI displace radiologists, or will it become a tool that makes them more productive?",
    "How will AI change the number of jobs in software engineering?",
    "Will AI reduce the need for human financial analysts and portfolio managers?",
    "How will AI affect the demand for human journalists and reporters?",
    "Will AI replace most customer service representatives?",
    "How will AI affect employment for truck drivers and delivery workers?",
    "Will AI displace human teachers, or augment them?",
    "How will AI affect the demand for management consultants?",
    "Will AI replace most data entry and bookkeeping jobs?",
    "How will AI change the demand for human graphic designers?",
    "Will AI reduce employment among research scientists?",
    "How will AI affect hiring for human resources professionals?",

    # --- Direct displacement: geographic variation ---
    "Will AI displacement affect the United States and Europe differently?",
    "How will AI affect employment in countries whose economies depend on call centers and BPO?",
    "Will AI widen or narrow the economic gap between developed and developing nations?",
    "How will AI displacement play out differently in China versus the United States?",
    "Will AI automation hit small-town economies harder than urban economies?",
    "How will AI affect jobs in countries with young, fast-growing populations like Nigeria and India?",
    "Will rural communities experience AI displacement differently than cities?",
    "How will AI affect labor markets in countries with weak social safety nets?",
    "Will AI create new tech hubs or further concentrate jobs in existing ones?",

    # --- Direct displacement: demographic impacts ---
    "Will AI displacement disproportionately affect women or men?",
    "How will AI affect employment prospects for workers without college degrees?",
    "Will older workers face greater displacement from AI than younger workers?",
    "How will AI affect employment opportunities for immigrants?",
    "Will AI widen or narrow the racial wealth gap through its effects on employment?",
    "How will AI displacement affect workers with disabilities?",
    "Will AI disproportionately displace workers in lower-income brackets?",
    "How will AI affect the employment prospects of people who are mid-career and overspecialized?",

    # --- Direct displacement: job creation vs. destruction ---
    "What new categories of jobs will AI create that don't exist today?",
    "Will AI create enough high-quality jobs to replace the ones it destroys?",
    "Will AI-created jobs require significantly higher skills than the jobs they replace?",
    "How long will it take for new AI-created jobs to absorb workers displaced from old ones?",
    "Will the new jobs AI creates pay as well as the ones it eliminates?",
    "Will AI create more jobs in developing or developed economies?",
    "Is the idea that AI will create more jobs than it destroys based on evidence or historical analogy?",
    "Will AI-created jobs be concentrated among the already-privileged?",

    # --- Direct displacement: augmentation vs. replacement ---
    "Will AI mostly augment human workers or replace them outright?",
    "Is the distinction between AI augmentation and AI replacement meaningful for displaced workers?",
    "Will jobs that start as human-AI collaboration eventually become AI-only?",
    "How long do human-AI hybrid roles typically last before full automation?",
    "Will AI augmentation raise the productivity bar so high that fewer workers are needed?",
    "Do AI copilot tools make junior workers more productive or make junior workers unnecessary?",
    "Will AI augmentation lead companies to hire fewer people at higher salaries or more people at lower salaries?",

    # --- Direct displacement: task vs. job displacement ---
    "Is it more accurate to say AI displaces tasks within jobs rather than entire jobs?",
    "If AI automates 40% of a job's tasks, does the job survive or get eliminated?",
    "Will partial task automation lead to job redesign or job cuts?",
    "How will employers respond when AI can do most but not all of what a worker does?",
    "Will task-level automation make remaining human tasks more or less satisfying?",

    # --- Direct displacement: transition dynamics ---
    "Can displaced workers realistically retrain for AI-economy jobs?",
    "How long will the transition period of AI-driven job displacement last?",
    "Will there be a generation of workers permanently left behind by AI automation?",
    "What happens to communities built around industries that AI makes obsolete?",
    "Will AI displacement create a permanent underclass of unemployable workers?",
    "How will displaced workers support themselves during the transition to an AI economy?",
    "Will the labor market reach a new equilibrium after AI disruption, or is this a permanent shift?",

    # --- Direct displacement: conditional & scenario-based ---
    "If AI reaches human-level performance on most cognitive tasks, what jobs remain?",
    "What happens to employment if AI progress plateaus at current capability levels?",
    "How would a sudden breakthrough in robotics change the displacement timeline?",
    "If open-source AI democratizes access, will that increase or decrease displacement?",
    "What would happen to employment if AI development were paused for five years?",
    "How would the displacement picture change if AI energy costs remain high?",
    "If AI can pass professional licensing exams, should licensed professionals worry about displacement?",

    # --- Direct displacement: measurement & evidence ---
    "How should we measure whether AI is actually displacing jobs or just changing them?",
    "What leading indicators would show AI displacement is accelerating?",

    # --- Economic structure: capital vs. labor share ---
    "How will AI change the distribution of wealth between capital and labor?",
    "Will AI-driven productivity gains translate into higher wages for workers?",
    "Will AI accelerate the decades-long decline of labor's share of national income?",
    "How will AI affect the bargaining power of individual workers versus employers?",
    "Will AI profits flow primarily to shareholders or be reinvested in ways that benefit workers?",
    "If AI doubles corporate productivity, who captures most of the surplus — workers, consumers, or shareholders?",
    "Will AI make capital investment more important than human capital for economic growth?",
    "How will AI affect the return on capital versus the return on labor?",
    "Will AI make it possible for a small number of firms to capture most economic value in an industry?",
    "Does AI fundamentally change the relationship between productivity growth and wage growth?",

    # --- Economic structure: inequality & distribution ---
    "Will AI increase or decrease economic inequality in developed nations?",
    "How will AI affect wage growth for median workers over the next 20 years?",
    "Will AI create a two-tier economy of AI-complementary and AI-displaced workers?",
    "How will AI affect the income gap between the top 10% and bottom 50% of earners?",
    "Will AI make inherited wealth more or less important for economic success?",
    "How will AI affect economic mobility — can a poor person still work their way up?",
    "Will AI widen the gap between high-skill and low-skill wages?",
    "How will AI affect the middle class — will it grow, shrink, or bifurcate?",
    "Will AI-driven inequality be primarily within countries or between countries?",
    "Is there a level of AI-driven inequality that becomes economically destabilizing?",
    "Will AI compress or expand the wage premium for a college education?",
    "How will AI affect the economic returns to different types of education (STEM vs. humanities vs. trades)?",

    # --- Economic structure: market structure & competition ---
    "Will AI lead to greater market concentration or more competition?",
    "How will AI change the relative economic power of large vs. small firms?",
    "Will AI lower barriers to entry for startups or raise them?",
    "How will AI affect the number of viable competitors in typical industries?",
    "Will AI enable monopolies or natural monopolies in new sectors?",
    "Can small businesses compete with large corporations that have superior AI capabilities?",
    "Will AI-driven network effects lead to winner-take-all markets?",
    "How will AI affect the balance of power between platforms and the businesses that depend on them?",
    "Will AI make it easier or harder for new firms to challenge incumbents?",
    "Will open-source AI level the playing field between large and small companies?",
    "How will AI affect the optimal size of firms — will companies get bigger or smaller?",

    # --- Economic structure: productivity & growth ---
    "Will AI deliver a sustained productivity boom comparable to electrification or the internet?",
    "How will AI affect GDP growth rates in advanced economies over the next two decades?",
    "Will AI-driven productivity gains show up in official economic statistics, or is there a measurement problem?",
    "Is AI more likely to produce one-time efficiency gains or compounding productivity growth?",
    "Will the productivity benefits of AI be broadly distributed or concentrated in a few sectors?",
    "How will AI affect the productivity gap between leading firms and average firms?",
    "Will AI reverse the productivity slowdown that advanced economies have experienced since the 1970s?",
    "Can AI-driven productivity growth outpace the economic costs of displacement?",
    "Will AI make economic output less dependent on the number of workers in the labor force?",
    "How will AI affect total factor productivity versus labor productivity?",

    # --- Economic structure: wages & compensation ---
    "Will AI put downward pressure on wages across the economy or only in specific sectors?",
    "How will AI affect the relationship between worker productivity and worker pay?",
    "Will AI lead to wage stagnation for the majority even as GDP grows?",
    "How will AI affect compensation structures — more performance-based or more flat?",
    "Will AI make geographic location less important for determining wages?",
    "Will AI narrow or widen the wage gap between countries?",
    "How will AI affect wages in sectors that are not directly automated?",
    "Will AI create a class of extremely high earners while compressing wages for everyone else?",
    "If AI makes most workers more productive, will competition for workers bid wages up or will the surplus go to employers?",

    # --- Economic structure: labor market structure ---
    "How will AI affect the gig economy and freelance work?",
    "Will AI shift more workers from full-time employment to contract and gig work?",
    "How will AI change the typical length of a career at a single company?",
    "Will AI make labor markets more fluid or more rigid?",
    "How will AI affect the power dynamic between employers and employees?",
    "Will AI-driven monitoring and management tools make workplaces more authoritarian?",
    "How will AI affect the demand for part-time versus full-time workers?",
    "Will AI lead to shorter working hours or longer ones?",
    "How will AI change the geographic distribution of economic activity — more concentrated or more dispersed?",
    "Will remote work enabled by AI tools reduce or increase wage inequality?",

    # --- Economic structure: global economic structure ---
    "Will AI make it easier or harder for developing nations to industrialize?",
    "How will AI affect the economic relationship between the Global North and Global South?",
    "Will AI reduce the advantage of cheap labor in global supply chains?",
    "How will AI affect international trade patterns?",
    "Will AI enable developing countries to leapfrog industrial stages, or will it lock in current hierarchies?",
    "How will AI affect the economic power of resource-rich versus technology-rich nations?",
    "Will AI make global supply chains more concentrated or more distributed?",
    "How will AI affect the viability of export-led growth strategies for developing economies?",
    "Will AI-driven automation bring manufacturing back to high-wage countries?",
    "How will AI change the economics of offshoring and nearshoring?",

    # --- Economic structure: structural transformation ---
    "Will AI cause a structural shift comparable to the move from agriculture to manufacturing?",
    "How will AI affect the service sector, which employs the majority of workers in developed economies?",
    "Will AI create entirely new economic sectors, or mostly transform existing ones?",
    "Is AI more likely to change what work gets done or how existing work gets done?",
    "How will AI affect the informal economy in developing countries?",
    "Will AI accelerate the shift toward a knowledge economy, or make knowledge work less valuable?",
    "How will AI change the economic role of the public sector?",
    "Will AI lead to economic degrowth in some sectors while others boom?",

    # --- Policy and regulation: income support & safety nets ---
    "Should governments implement UBI in response to AI automation?",
    "Is universal basic income the best response to AI displacement, or are there better alternatives?",
    "Should UBI payments be funded specifically by taxing AI company profits?",
    "How should unemployment insurance be redesigned for an era of AI-driven job loss?",
    "Should workers displaced by AI receive longer or more generous unemployment benefits than those displaced by other causes?",
    "Is a negative income tax more practical than UBI for addressing AI-driven inequality?",
    "Should governments guarantee a minimum income floor regardless of employment status in an AI economy?",
    "Will expanding the social safety net reduce the incentive to adapt to AI, or is that concern overstated?",
    "Should disability benefits be expanded to cover workers whose skills have been made obsolete by AI?",

    # --- Policy and regulation: taxation & fiscal policy ---
    "What tax policies should be adopted to address AI-driven inequality?",
    "Should there be a tax on AI and robots that replace human workers?",
    "Should AI-generated output be taxed differently than human-generated output?",
    "How should capital gains taxes change if AI dramatically increases returns to capital?",
    "Should companies receive tax breaks for using AI to augment workers rather than replace them?",
    "Is a windfall profits tax on AI companies justified?",
    "How should governments tax the productivity gains from AI when those gains don't show up as wages?",
    "Should there be a payroll tax equivalent applied to AI systems that perform work previously done by humans?",
    "Will AI-driven automation erode the tax base by reducing the number of wage earners?",
    "Should sovereign wealth funds be created from AI taxation to benefit all citizens?",

    # --- Policy and regulation: education & retraining ---
    "Is retraining sufficient to address AI-driven job displacement?",
    "Should governments mandate employer-funded retraining when companies automate jobs?",
    "How should public education systems change to prepare students for an AI-transformed economy?",
    "Are coding bootcamps and short retraining programs realistic solutions for displaced workers?",
    "Should the government subsidize lifelong learning accounts for all workers?",
    "Who should bear the cost of retraining — workers, employers, or the public?",
    "How should vocational education adapt when even skilled trades may be affected by AI?",
    "Should universities restructure curricula in response to AI, and if so, how?",
    "Is the expectation that displaced workers will retrain a realistic policy or a politically convenient fiction?",
    "How effective are past government retraining programs as models for AI displacement?",
    "Should retraining programs focus on AI-complementary skills or AI-resistant occupations?",

    # --- Policy and regulation: labor law & worker protection ---
    "Should AI deployment in workplaces be regulated to protect workers?",
    "Should companies be required to provide transition support when automating jobs?",
    "What role should labor unions play in the age of AI?",
    "Should workers have the right to be informed before their job is automated?",
    "Should companies need government approval before automating jobs above a certain scale?",
    "How should labor law adapt to protect workers who collaborate with AI systems?",
    "Should there be mandatory transition periods before companies can fully automate a role?",
    "Do existing employment discrimination laws adequately address AI-driven hiring and firing?",
    "Should workers have a legal right to challenge decisions made by AI systems in the workplace?",
    "How should workplace surveillance laws change as AI monitoring becomes more pervasive?",
    "Should non-compete agreements be banned to help displaced workers find new employment?",
    "Should gig workers who are managed by AI algorithms have the same protections as full-time employees?",

    # --- Policy and regulation: trade & industrial policy ---
    "Should countries use industrial policy to ensure they lead in AI development?",
    "How should trade agreements address the displacement effects of AI across borders?",
    "Should developing countries restrict AI imports to protect domestic employment?",
    "Will AI make tariffs and trade barriers more or less effective as economic tools?",
    "Should governments subsidize domestic AI development or focus on managing its impacts?",
    "How should export controls on AI technology balance innovation against displacement in other countries?",
    "Should there be international agreements on managing AI's labor market effects?",
    "Is protectionism a rational response to AI-driven job displacement?",

    # --- Policy and regulation: antitrust & competition policy ---
    "How should antitrust policy adapt to AI-driven market concentration?",
    "Should AI companies that dominate labor markets face different antitrust scrutiny than traditional monopolies?",
    "How should regulators address AI-driven market concentration without stifling innovation?",
    "Should data used to train AI models be treated as a public resource?",
    "Do current antitrust frameworks adequately address the competitive dynamics of AI?",
    "Should regulators force interoperability between AI systems to prevent lock-in?",
    "How should competition policy handle AI companies that operate as both platforms and competitors?",

    # --- Policy and regulation: pace & approach to regulation ---
    "Should AI regulation prioritize speed of deployment or caution about labor impacts?",
    "Is it better to regulate AI proactively or wait until harms are clearly demonstrated?",
    "Should AI labor regulation be handled at the national, state, or international level?",
    "Can regulatory sandboxes effectively balance AI innovation with worker protection?",
    "Should AI regulation differ by sector based on the sensitivity of potential displacement?",
    "How should governments regulate AI when the technology is advancing faster than policy can adapt?",
    "Is self-regulation by AI companies a credible alternative to government regulation?",
    "Should there be a moratorium on AI deployment in high-employment sectors until impact studies are completed?",
    "How should regulatory approaches differ between AI that augments workers and AI that replaces them?",

    # --- Policy and regulation: corporate governance & responsibility ---
    "Should companies be required to report the number of jobs displaced by AI adoption?",
    "Should corporate boards include worker representatives when making AI adoption decisions?",
    "How should executive compensation be structured when AI displaces significant numbers of workers?",
    "Should companies that profit from AI automation be required to fund community transition programs?",
    "Do shareholders have a responsibility to consider displacement effects when AI boosts profits?",
    "Should ESG frameworks include metrics on how companies manage AI-driven displacement?",
    "Should companies be required to conduct employment impact assessments before deploying AI?",

    # --- Policy and regulation: international & comparative policy ---
    "Which country's approach to AI and labor policy is most likely to succeed — the US, EU, or China?",
    "Should international institutions like the ILO set standards for managing AI displacement?",
    "How should developing countries regulate AI given that they lack the fiscal resources for large safety nets?",
    "Is the EU's precautionary approach to AI regulation better or worse for workers than the US approach?",
    "Should there be a global AI governance body focused on labor impacts?",
    "How do different cultural attitudes toward work affect the policy response to AI displacement?",
    "Can countries with strong social democracies (Nordics) handle AI displacement better than liberal market economies?",

    # --- Policy and regulation: political economy ---
    "Will AI displacement create political instability or populist backlash?",
    "How will AI-driven inequality affect democratic governance?",
    "Will the economic losers from AI automation have enough political power to demand protections?",
    "Is there a risk that AI policy will be captured by the companies it's meant to regulate?",
    "How should politicians communicate about AI displacement without causing panic or complacency?",
    "Will AI displacement be a defining political issue of the next decade?",
    "Can democratic institutions respond quickly enough to manage AI-driven economic disruption?",
    "Will AI-driven inequality strengthen the case for socialism or for free-market capitalism?",
    "How will lobbying by AI companies shape the policy response to displacement?",
    "Is the political will to address AI displacement realistic given other competing priorities?",

    # --- Historical analogies: specific historical episodes ---
    "Is AI automation comparable to the Industrial Revolution's impact on jobs?",
    "What can the history of agricultural mechanization teach us about AI displacement?",
    "What lessons from the automation of manufacturing apply to AI and white-collar work?",
    "What parallels exist between AI displacement and the decline of handloom weavers during industrialization?",
    "Did the introduction of the printing press destroy more jobs than it created?",
    "How did the mechanization of agriculture in the early 20th century affect rural communities, and what does that suggest about AI?",
    "What happened to bank tellers after ATMs were introduced, and is that pattern likely to repeat with AI?",
    "How did the introduction of containerized shipping reshape employment, and does AI pose similar structural risks?",
    "What can the decline of the US steel industry teach us about managing AI-driven displacement?",
    "How did the shift from horse-drawn transport to automobiles affect employment, and what are the parallels to AI?",
    "What happened to telephone operators when automatic switching was introduced?",
    "How did the mechanization of coal mining affect mining communities, and what lessons apply to AI?",
    "What can the history of textile automation in 19th-century England tell us about the social costs of rapid technological change?",
    "How did the introduction of spreadsheet software affect the accounting profession?",
    "What happened to travel agents after online booking platforms emerged?",
    "How did the introduction of word processors affect secretarial and typing pool employment?",
    "What can the history of elevator operators tell us about how quickly automated systems replace human workers?",
    "How did the Green Revolution's agricultural technologies affect employment in developing countries?",

    # --- Historical analogies: the Industrial Revolution as analogy ---
    "Did workers benefit from the Industrial Revolution within their lifetimes, or only in subsequent generations?",
    "How long did it take for real wages to rise after the start of the Industrial Revolution?",
    "Were the social upheavals of the Industrial Revolution an acceptable cost of long-term economic growth?",
    "Did the factory system create better or worse working conditions than the cottage industries it replaced?",
    "How did the Industrial Revolution affect economic inequality in its first 50 years?",
    "Is the comparison between AI and the Industrial Revolution reassuring or alarming when you look at the actual history?",
    "What role did labor organizing and political reform play in ensuring workers eventually benefited from industrialization?",
    "Would the Industrial Revolution have been less painful with better social safety nets, and does that lesson apply to AI?",

    # --- Historical analogies: the computer revolution as analogy ---
    "Did the personal computer revolution of the 1980s and 1990s cause net job loss or net job creation?",
    "How did the rise of the internet change employment patterns, and is AI likely to follow a similar path?",
    "What happened to productivity and wages during the computer revolution, and does that predict AI's impact?",
    "Why didn't the IT revolution of the 1990s cause the mass unemployment that some predicted?",
    "Did the dot-com boom and bust offer lessons for how AI investment cycles might affect employment?",
    "How did the automation of manufacturing through robotics in the 1980s affect factory employment?",
    "Did e-commerce destroy more retail jobs than it created in logistics and tech?",
    "How did the introduction of AI-powered search engines affect information work?",

    # --- Historical analogies: where analogies break down ---
    "How does AI differ from previous waves of technological automation?",
    "What makes AI fundamentally different from every previous automation technology?",
    "Is the argument that technology always creates more jobs based on a pattern that could break?",
    "Can we trust historical patterns when AI is the first technology that automates cognitive work at scale?",
    "Does the speed of AI adoption invalidate comparisons to slower technological transitions?",
    "Are historical analogies misleading because past automation affected one sector at a time while AI affects many simultaneously?",
    "Does AI's ability to learn and improve autonomously make it categorically different from previous tools?",
    "Is it fair to compare AI to past technologies when past technologies augmented physical labor and AI augments mental labor?",
    "Does the global interconnectedness of modern economies make AI displacement harder to manage than past transitions?",
    "Are historical employment statistics reliable enough to draw conclusions about past technological transitions?",
    "Could AI be the technology that finally makes the Luddites right?",

    # --- Historical analogies: mechanisms of historical adjustment ---
    "How did new industries historically emerge to absorb workers displaced by technology?",
    "What role did geographic mobility play in past labor market adjustments to technology?",
    "How long did past technological transitions take to reach a new labor market equilibrium?",
    "Did government intervention help or hinder labor market adjustment during past technological transitions?",
    "What role did education expansion play in helping workers adapt to past waves of automation?",
    "How did entrepreneurship contribute to job creation during past technological disruptions?",
    "Were past transitions smoother in countries with strong institutions versus those without?",
    "How did demographic changes interact with technological displacement in past transitions?",
    "What role did infrastructure investment play in creating new employment during past technological shifts?",

    # --- Historical analogies: predictions and track records ---
    "Did past predictions about technology destroying jobs prove accurate?",
    "How does the pace of AI adoption compare to previous technological transitions?",
    "How accurate have economists been at predicting the employment effects of new technologies?",
    "Did Keynes's prediction of widespread technological unemployment by 2030 prove correct?",
    "Why did predictions of mass unemployment from 1960s automation fail to materialize?",
    "Were the optimistic predictions about technology and jobs more or less accurate than pessimistic ones?",
    "What do the track records of past technology forecasters suggest about current AI predictions?",
    "Have labor economists consistently underestimated or overestimated the job-creating effects of new technology?",
    "Why do predictions about technology and employment tend to be wrong, and does that pattern apply to AI?",

    # --- Historical analogies: transition costs and who bears them ---
    "Who bore the greatest costs during the Industrial Revolution — and is that pattern likely to repeat with AI?",
    "How did the decline of the American Rust Belt affect workers who couldn't relocate?",
    "Did the workers displaced by agricultural mechanization ever recover economically?",
    "What happened to communities built around industries that were displaced by technology in the past?",
    "How many generations did it take for the losers of past technological transitions to recover?",
    "Were the transition costs of past automation distributed fairly across society?",
    "Did past technological transitions increase or decrease regional economic inequality?",

    # --- Historical analogies: lessons and takeaways ---
    "Is the Luddite fallacy still a fallacy when it comes to AI?",
    "What is the single most important lesson from past technological transitions for managing AI?",
    "If you could apply one historical policy response to AI displacement, which would it be?",
    "Does history suggest that the benefits of transformative technology eventually reach everyone, or only the majority?",
    "What historical mistakes should policymakers avoid when managing AI's labor market effects?",
    "Does history support the optimistic or pessimistic view of AI and employment?",
    "Should we trust historical patterns or treat AI as genuinely unprecedented?",
    "What would a historian in 2075 say about how we handled AI displacement?",
    "Is the belief that technology always creates net employment a well-supported historical claim or survivorship bias?",
    "What is the most overlooked historical analogy for understanding AI's economic impact?",

    # --- Individual career: career choice & education ---
    "Should a college student avoid majoring in fields AI might automate?",
    "Is learning to code still a good career investment given AI coding tools?",
    "Should students choose their major based on what AI is least likely to automate?",
    "Is a four-year college degree more or less valuable in an AI economy?",
    "Should young people pursue trades instead of college given AI's impact on knowledge work?",
    "Is a graduate degree in a specialized field a good hedge against AI or a risky bet on a shrinking niche?",
    "Should students prioritize STEM education or liberal arts in an AI-dominated economy?",
    "Is it smarter to specialize deeply or be a generalist in an AI economy?",
    "Should aspiring lawyers reconsider their career given AI's impact on legal work?",
    "Is medical school still a good investment if AI can match diagnostic accuracy?",
    "Should young people entering the workforce today expect to change careers multiple times due to AI?",
    "Is an MBA still valuable if AI can perform much of what management consultants do?",

    # --- Individual career: skills & adaptability ---
    "What skills should workers develop to remain competitive in an AI economy?",
    "What cognitive skills will be most valuable when AI handles routine analytical work?",
    "Are soft skills like communication and empathy genuinely AI-proof, or is that wishful thinking?",
    "Should workers invest in learning to use AI tools, or in developing skills AI can't replicate?",
    "How important is creativity as a career differentiator in an AI economy?",
    "Will emotional intelligence become the most important career skill as AI advances?",
    "Is continuous learning a realistic expectation for most workers, or an unreasonable burden?",
    "Should workers focus on becoming AI-literate or on deepening non-AI expertise?",
    "How valuable is domain expertise when AI can rapidly absorb domain knowledge?",
    "Will the ability to manage and direct AI systems be a widely available skill or a rare one?",
    "Is entrepreneurship a more reliable career path than employment in an AI economy?",

    # --- Individual career: mid-career workers ---
    "How should a 40-year-old factory worker prepare for AI automation?",
    "How should a 50-year-old office worker respond to AI tools that can do most of their job?",
    "Is it realistic for mid-career professionals to retrain for AI-economy jobs?",
    "Should mid-career workers accept lower-paying roles if their current job is at risk from AI?",
    "How should a mid-career worker evaluate whether their industry will be disrupted by AI?",
    "Is it better to double down on existing expertise or pivot to a new field when AI threatens your job?",
    "Should workers in their 40s and 50s prioritize job security or adaptability?",
    "How should a mid-career professional in media or journalism respond to AI content generation?",
    "What options does a mid-career accountant have if AI automates most of their work?",

    # --- Individual career: blue-collar & service workers ---
    "How should truck drivers prepare for the possibility of autonomous vehicles?",
    "Should warehouse workers worry about AI and robotics replacing their jobs?",
    "How should retail workers think about their career prospects as AI transforms commerce?",
    "What career options exist for factory workers whose jobs are automated by AI?",
    "Should food service workers view AI automation as an immediate threat or a distant possibility?",
    "How should construction workers think about AI and automation risks in their industry?",
    "Is the advice to retrain for tech jobs realistic for blue-collar workers?",
    "Should tradespeople like electricians and plumbers feel secure from AI displacement?",

    # --- Individual career: geographic & economic context ---
    "Should workers in AI hub cities (San Francisco, London) feel more or less secure than those elsewhere?",
    "How should workers in developing countries prepare for AI's impact on outsourced jobs?",
    "Should workers consider relocating to regions with more AI-resilient job markets?",
    "How should workers in rural areas with limited job markets prepare for AI disruption?",
    "Is remote work an opportunity or a threat for workers in areas with lower AI adoption?",
    "Should workers in small towns invest in digital skills or double down on local, physical-presence jobs?",

    # --- Individual career: financial planning & risk ---
    "How should individuals adjust their financial planning for the possibility of AI-driven job loss?",
    "Should workers save more aggressively as a hedge against AI displacement?",
    "Is investing in AI companies a good personal financial strategy even if AI threatens your job?",
    "How should retirement planning change if AI may shorten your career?",
    "Should workers prioritize paying off debt or building savings given AI displacement risk?",
    "Is homeownership riskier in an era of potential AI-driven job instability?",
    "How should dual-income households think about AI risk if both careers are vulnerable?",

    # --- Individual career: freelancers & entrepreneurs ---
    "Is it worth investing in a career in AI if the field might automate itself?",
    "Should freelancers embrace AI tools that could eventually replace them?",
    "Is starting a business that leverages AI a good career move, or will AI commoditize the advantage?",
    "How should freelance writers, designers, and programmers adapt to AI competition?",
    "Will AI make it easier or harder to succeed as a solo entrepreneur?",
    "Should freelancers specialize in AI-proof niches or become AI-augmented generalists?",
    "How should consultants adapt their practice when clients can get AI-generated advice for free?",

    # --- Individual career: workplace dynamics ---
    "Should young professionals prefer jobs that involve human interaction over analytical work?",
    "Should you tell your employer that AI could automate your role?",
    "How should workers respond when their company starts piloting AI tools for their function?",
    "Is it better to lead AI adoption in your team or resist it to protect your position?",
    "Should workers negotiate for retraining clauses in their employment contracts?",
    "How should you position yourself at work if AI is automating parts of your job?",
    "Is volunteering to lead AI implementation at your company a career opportunity or a way to automate yourself?",
    "How should workers handle the anxiety and uncertainty of potential AI displacement?",

    # --- Individual career: philosophical & psychological ---
    "Should people define their identity and self-worth less through their career given AI's potential impact?",
    "How should parents advise their children about careers in an era of AI uncertainty?",
    "Is career anxiety about AI a rational response or media-driven panic?",
    "Should individuals feel personally responsible for adapting to AI, or is that a systemic problem?",
    "Is it possible to have a fulfilling career if AI handles the intellectually challenging parts of your work?",
    "How should workers balance optimism about AI's potential with prudent career planning?",
    "Does AI make the concept of a lifelong career obsolete?",
    "Should people pursue passion-driven careers assuming AI will handle economically productive work?",
    "Is the expectation to constantly adapt to new technology fair to workers?",
    "How much career disruption from AI should an individual be expected to absorb in a lifetime?",
    "Should workers view AI as a collaborator to embrace or a competitor to outperform?",
    "Is career resilience a personal virtue or a systemic responsibility?",

    # --- Sector-specific: healthcare ---
    "How will AI affect healthcare employment — net positive or negative?",
    "Will AI reduce or increase the number of nurses needed in hospitals?",
    "How will AI diagnostic tools change the role of primary care physicians?",
    "Will AI-powered drug discovery create or eliminate pharmaceutical research jobs?",
    "How will AI affect mental health professions — will therapy be automated?",
    "Will AI reduce healthcare costs enough to expand the sector and create net new jobs?",
    "How will AI affect the demand for medical specialists versus generalists?",
    "Will AI shift healthcare employment from clinical roles to data and technology roles?",
    "How will AI affect healthcare employment in developing countries with doctor shortages?",

    # --- Sector-specific: legal profession ---
    "How will AI change employment in the legal profession?",
    "Will AI make legal services more accessible and thus increase demand for lawyers?",
    "How will AI affect the economics of large law firms versus solo practitioners?",
    "Will AI-driven contract analysis eliminate more junior or senior legal positions?",
    "How will AI change the role of judges and courtroom proceedings?",
    "Will AI reduce the cost of legal services enough to expand the market and create net employment?",
    "How will AI affect compliance and regulatory jobs in the legal sector?",

    # --- Sector-specific: financial services ---
    "Will AI increase or decrease employment in the financial services sector?",
    "Will AI-driven algorithmic trading eliminate most human trading jobs?",
    "How will AI affect employment in insurance underwriting and claims processing?",
    "Will AI replace or augment financial advisors who serve retail clients?",
    "How will AI affect employment in banking — from tellers to back-office operations?",
    "Will AI in fintech create enough new jobs to offset losses in traditional finance?",
    "How will AI change the skills required for careers in private equity and venture capital?",

    # --- Sector-specific: creative industries ---
    "What is AI's likely impact on creative industries (writing, art, music)?",
    "Will AI-generated art, music, and writing replace human creators or expand the market?",
    "How will AI affect employment for professional photographers and videographers?",
    "Will AI tools democratize creative work or devalue it?",
    "How will AI change the economics of the music industry for working musicians?",
    "Will AI-generated content flood markets and drive down prices for human-created work?",
    "How will AI affect employment in advertising and marketing creative roles?",
    "Will AI make it easier or harder for independent artists to earn a living?",
    "How will AI affect the film and television industry's workforce?",

    # --- Sector-specific: education ---
    "Will AI help or hurt employment in the education sector?",
    "Will AI tutoring systems reduce the demand for human teachers?",
    "How will AI change the role of university professors?",
    "Will AI make education more personalized and thus require more or fewer educators?",
    "How will AI affect employment in corporate training and professional development?",
    "Will AI-driven online education reduce employment in traditional educational institutions?",
    "How will AI change the demand for school administrators and support staff?",

    # --- Sector-specific: transportation & logistics ---
    "How will autonomous vehicles affect transportation jobs and the broader economy?",
    "How quickly will autonomous trucks displace long-haul trucking jobs?",
    "Will AI-optimized logistics create enough new roles to offset warehouse automation?",
    "How will AI affect employment for taxi and rideshare drivers?",
    "Will autonomous shipping reduce employment in the maritime industry?",
    "How will AI affect employment in air traffic control and aviation?",
    "Will last-mile delivery automation eliminate delivery driver jobs?",

    # --- Sector-specific: manufacturing ---
    "How will AI-powered robotics change the remaining manufacturing jobs in developed countries?",
    "Will AI make reshoring of manufacturing viable and create new factory jobs?",
    "How will AI affect quality control and inspection jobs in manufacturing?",
    "Will AI-driven predictive maintenance reduce or change the nature of manufacturing maintenance jobs?",
    "How will AI affect the demand for manufacturing engineers versus line workers?",
    "Will smart factories require more or fewer workers than traditional ones?",

    # --- Sector-specific: technology sector ---
    "Will AI automate enough software engineering to reduce tech employment?",
    "How will AI affect employment for data scientists and data analysts?",
    "Will AI create more cybersecurity jobs than it eliminates in other tech roles?",
    "How will AI change employment in IT support and system administration?",
    "Will the AI industry itself create enough jobs to offset AI-driven losses in other tech roles?",
    "How will AI affect employment for product managers and project managers in tech?",

    # --- Sector-specific: government & public sector ---
    "How will AI affect employment in government administration and bureaucracy?",
    "Will AI reduce the need for human workers in tax collection and social services?",
    "How will AI change employment in law enforcement and public safety?",
    "Will AI reduce employment in the military, or shift it toward different roles?",
    "How will AI affect jobs in intelligence analysis and national security?",
    "Will AI in public services free up government workers for higher-value tasks or eliminate their positions?",

    # --- Sector-specific: energy & environment ---
    "How will AI affect employment in the oil and gas industry?",
    "Will AI accelerate the clean energy transition in ways that create or destroy jobs?",
    "How will AI change employment in mining and resource extraction?",
    "Will AI-optimized energy grids reduce or change the nature of utility employment?",
    "How will AI affect jobs in environmental monitoring and conservation?",

    # --- Sector-specific: real estate & construction ---
    "How will AI affect employment for real estate agents and brokers?",
    "Will AI-driven property valuation and matching reduce the need for human intermediaries?",
    "How will AI and robotics change employment in the construction industry?",
    "Will AI-assisted design reduce the demand for architects and engineers?",

    # --- Sector-specific: media & journalism ---
    "Will AI-generated news articles replace human journalists?",
    "How will AI affect employment in publishing — from editors to marketers?",
    "Will AI change the economics of local journalism and small media outlets?",
    "How will AI affect employment in public relations and communications?",

    # --- Sector-specific: agriculture ---
    "How will AI affect agricultural employment in developing countries?",
    "How will AI-driven precision agriculture change farm employment in developed countries?",
    "Will AI and robotics make small-scale farming more or less viable?",
    "How will AI affect employment in the food processing and distribution chain?",

    # --- Sector-specific: science & research ---
    "Will AI accelerate scientific research enough to create net new research positions?",
    "How will AI change the role of laboratory technicians and research assistants?",
    "Will AI-driven research reduce the value of human scientific expertise?",
    "How will AI affect employment in clinical trials and pharmaceutical development?",

    # --- Sector-specific: retail & customer service ---
    "What will AI mean for retail and customer service jobs?",

    # --- Sector-specific: cross-sector patterns ---
    "Which economic sector will be most transformed by AI in the next decade?",
    "Are there sectors that are genuinely immune to AI disruption?",
    "Will AI cause employment to shift between sectors or shrink across all of them?",
    "How will AI affect the relative size of the service sector versus the goods-producing sector?",

    # --- Philosophical / values: justice & fairness ---
    "Is it morally acceptable to automate jobs if it increases aggregate GDP?",
    "How should we weigh aggregate prosperity against distributional fairness in AI policy?",
    "Is AI-driven displacement a form of injustice, or just an unfortunate side effect of progress?",
    "Do the people who benefit most from AI have a moral obligation to compensate those who lose?",
    "Is it fair that AI developers and investors capture most of the economic value while workers bear the displacement costs?",
    "Should the moral evaluation of AI automation depend on whether displaced workers find new jobs?",
    "Is there a difference between displacing workers through AI and displacing them through offshoring — morally?",
    "Does the principle of do no harm apply to companies automating jobs?",
    "Is it just to allow some people to become enormously wealthy from AI while others lose their livelihoods?",
    "Should AI's impact on the worst-off members of society be the primary moral consideration?",
    "Is a society that maximizes GDP through AI but tolerates high unemployment morally acceptable?",
    "Does fairness require that AI's productivity gains be distributed to everyone, or only to those who contributed?",

    # --- Philosophical / values: human dignity & work ---
    "Is meaningful work a human right that AI threatens?",
    "Is there an inherent dignity in work that is lost when AI replaces human effort?",
    "Does AI threaten human dignity by making people economically unnecessary?",
    "Is being useful to others through work essential to human flourishing?",
    "Can a society maintain human dignity if most people don't need to work?",
    "Is the connection between work and self-worth a cultural construct that can change, or something deeper?",
    "Does AI automation risk creating a class of people who feel purposeless?",
    "Is it degrading to be replaced by a machine, or is that an outdated sentiment?",
    "Should we care about the subjective experience of displacement even if the economic outcomes are positive?",
    "Does the loss of craft and expertise to AI represent a loss of something morally valuable?",
    "Is there moral value in human effort even when AI can produce a better result?",

    # --- Philosophical / values: rights & obligations ---
    "Do workers have a right to protection from technological displacement?",
    "What obligations do AI companies have to workers displaced by their products?",
    "Do workers have a moral right to their jobs, or only to fair treatment during transitions?",
    "Should the right to meaningful employment be considered a fundamental human right?",
    "Do future generations have a right to employment opportunities, or should we optimize for their material welfare?",
    "What moral obligations do AI researchers have to consider the employment impact of their work?",
    "Do consumers have an ethical obligation to prefer human-made goods and services over AI-generated ones?",
    "Should companies have a moral duty to consider employment effects before deploying AI, even if not legally required?",
    "Do governments have a moral obligation to protect their citizens from AI displacement?",
    "Is there a moral right to not be managed, evaluated, or replaced by an algorithm?",

    # --- Philosophical / values: democratic values & power ---
    "Does concentrated AI ownership threaten democratic governance?",
    "Should citizens have a democratic say in how AI is deployed in the economy?",
    "Is it compatible with democracy for a small number of AI companies to control the economic fate of millions?",
    "Does AI-driven inequality undermine the political equality that democracy requires?",
    "Should AI deployment decisions that affect large numbers of workers require democratic consent?",
    "Is technocratic management of AI displacement compatible with democratic values?",
    "Does the power that AI gives to employers over workers conflict with democratic principles?",
    "Should the pace and direction of AI development be subject to public deliberation?",

    # --- Philosophical / values: intergenerational ethics ---
    "Is it ethical to accept short-term displacement for long-term economic gains that benefit future generations?",
    "Do we owe it to future generations to develop AI as fast as possible, even at the cost of current workers?",
    "Is it fair to impose transition costs on the current generation for the benefit of later ones?",
    "How should we weigh the welfare of people alive today against the potential prosperity AI could create for future generations?",
    "Should parents sacrifice their own economic security to prepare their children for an AI economy?",
    "Is it ethical to make irreversible AI deployment decisions when we don't know the long-term consequences?",

    # --- Philosophical / values: meaning, purpose & the good life ---
    "Should society prioritize economic efficiency or employment stability?",
    "If AI eliminates the need for most human labor, is that a utopia or a dystopia?",
    "Can humans find meaning and purpose without economic necessity driving them to work?",
    "Would a post-work society enabled by AI be liberating or psychologically destructive?",
    "Is leisure a genuine substitute for the sense of purpose that work provides?",
    "Should society invest in helping people find meaning outside of work as AI takes over more tasks?",
    "Does AI threaten the narrative of meritocracy — that hard work leads to success?",
    "Is a life of AI-enabled abundance without work a life worth living?",
    "Will AI force a philosophical reckoning with what humans are actually for?",
    "Should we mourn the loss of jobs that were tedious and unfulfilling, or does all work have value?",
    "Is the fear of purposelessness in a post-work world a real concern or a failure of imagination?",

    # --- Philosophical / values: responsibility & blame ---
    "Who is morally responsible when AI displaces workers — the developers, the deploying companies, or the market?",
    "Can we assign moral blame for AI displacement when no single actor intended the outcome?",
    "Is it morally different to automate a job knowing it will displace someone versus automating without knowing?",
    "Should AI companies be held morally accountable for displacement even when they're following market incentives?",
    "Is society collectively responsible for AI displacement, or is it the responsibility of specific actors?",
    "Do investors in AI companies bear moral responsibility for the displacement their investments enable?",

    # --- Philosophical / values: progress, technology & values ---
    "Is it ethical to slow AI progress to give workers time to adapt?",
    "Should the benefits of AI productivity be shared universally or accrue to innovators?",
    "Is technological progress inherently good, or must it be evaluated by its effects on people?",
    "Should we value economic efficiency over human welfare when the two conflict?",
    "Is the belief that technology always leads to progress a form of faith rather than evidence?",
    "Does AI force us to choose between economic growth and human well-being?",
    "Is resistance to AI displacement a valid moral stance or backward-looking nostalgia?",
    "Should moral philosophy play a larger role in AI development decisions?",
    "Is the pursuit of AI-driven efficiency a moral imperative or a moral hazard?",
    "Can a technology be morally neutral if its deployment predictably harms specific groups?",
    "Is optimism about AI and jobs a moral position or an empirical prediction?",
    "Should we evaluate AI's impact on employment by utilitarian, deontological, or virtue ethics standards?",
    "Is the moral case for AI automation stronger when it eliminates dangerous work versus comfortable work?",
    "Does the moral calculus of AI displacement change if the displaced workers are in rich versus poor countries?",
    "Is there a moral difference between AI that augments human capability and AI that replaces it entirely?",
    "Should moral intuitions about fairness constrain AI deployment even if the aggregate outcomes are positive?",

    # --- Cross-cutting: steelmanning & perspective-taking ---
    "What is the strongest argument against your position on AI and jobs?",
    "Where do optimists and pessimists about AI and jobs most agree?",
    "What is the strongest evidence-based case that AI will be net positive for employment?",
    "What is the strongest evidence-based case that AI will be net negative for employment?",
    "What would a reasonable optimist say to a reasonable skeptic about AI and jobs?",
    "What legitimate concerns do AI skeptics raise that optimists should take seriously?",
    "What legitimate points do AI optimists make that skeptics should acknowledge?",
    "If you had to argue the opposite of your position on AI and jobs, what would you say?",
    "What is the most common strawman argument made by AI optimists about skeptics?",
    "What is the most common strawman argument made by AI skeptics about optimists?",
    "Where is the strongest common ground between AI optimists and skeptics on economic policy?",
    "What would a fair-minded summary of both sides of the AI employment debate look like?",

    # --- Cross-cutting: epistemic humility & uncertainty ---
    "How confident should we be in any prediction about AI's effect on employment?",
    "What are the most important unknowns about AI's economic impact?",
    "What do we genuinely not know about AI's impact on employment?",
    "How much uncertainty should we expect in any prediction about AI and jobs?",
    "Is the honest answer to most questions about AI and employment simply we don't know yet?",
    "What would it take to resolve the disagreement between AI optimists and skeptics?",
    "Are there questions about AI and employment that may be unanswerable in advance?",
    "How should policymakers act under deep uncertainty about AI's employment effects?",
    "Is it more dangerous to overestimate or underestimate AI's impact on jobs?",
    "Should we plan for the worst-case scenario of AI displacement even if it's unlikely?",
    "How should the public interpret conflicting expert opinions about AI and employment?",
    "What are the limits of economic modeling for predicting AI's labor market impact?",
    "Is anyone actually qualified to predict how AI will affect employment in 20 years?",
    "How should we update our beliefs about AI and jobs as new evidence emerges?",

    # --- Cross-cutting: evidence evaluation ---
    "What evidence would change your mind about AI's economic impact?",
    "If AI does cause mass displacement, what is the earliest we would see clear evidence?",
    "What existing empirical evidence best supports the optimistic view of AI and employment?",
    "What existing empirical evidence best supports the pessimistic view of AI and employment?",
    "How reliable are current economic studies on AI's impact on jobs?",
    "Are labor economists or technologists better positioned to predict AI's employment effects?",
    "How should we weigh anecdotal evidence of AI displacement against aggregate employment data?",
    "Is the current evidence on AI and jobs strong enough to justify major policy changes?",
    "What kind of natural experiment would help us understand AI's true employment impact?",
    "How should we interpret the fact that unemployment hasn't spiked despite rapid AI adoption?",
    "Does the current labor market data tell us anything meaningful about AI's long-term impact?",
    "What data would we need to collect now to evaluate AI's employment impact in 10 years?",
    "Are surveys of business leaders about AI adoption plans reliable predictors of actual displacement?",
    "How should we weigh economic theory against empirical observation when they disagree about AI?",

    # --- Cross-cutting: scenario analysis ---
    "What does the best-case scenario for AI and employment look like in 2040?",
    "What does the worst-case scenario for AI and employment look like in 2040?",
    "What does the most likely scenario for AI and employment look like in 2040?",
    "If AI development accelerates beyond current expectations, how does the employment picture change?",
    "If AI development stalls or hits fundamental limits, what happens to the displacement debate?",
    "What happens to employment if AI becomes very good at physical tasks as well as cognitive ones?",
    "How does the employment outlook change if AI remains expensive versus becoming very cheap?",
    "What if AI displaces 30% of current jobs but creates 30% new ones — is that a crisis or a transition?",
    "What happens if AI displacement is concentrated in a few countries while others are largely unaffected?",
    "How would a major AI safety incident or failure change the trajectory of AI deployment and employment?",

    # --- Cross-cutting: framing & definitions ---
    "Is the debate about AI and jobs really about technology, or about how we distribute economic gains?",
    "Are we asking the right questions about AI and employment, or are we missing the bigger picture?",
    "Is the focus on job counts the right way to think about AI's impact, or should we focus on job quality?",
    "Does the framing of AI as taking jobs misrepresent what's actually happening in labor markets?",
    "Is the AI employment debate fundamentally an economic question or a political one?",
    "Should we think about AI displacement in terms of individuals or in terms of systems?",
    "Is the distinction between AI augmentation and AI replacement a useful framework or a false dichotomy?",
    "Are we right to focus on employment effects when AI may change the nature of work itself?",
    "Is technological unemployment the right concept for understanding AI's impact, or do we need new frameworks?",
    "Should the AI employment debate focus more on transition dynamics and less on end states?",

    # --- Cross-cutting: synthesis & integration ---
    "How do the economic, ethical, and political dimensions of AI displacement interact?",
    "Can we have both rapid AI progress and broad-based economic security?",
    "Is there a policy framework that could satisfy both AI optimists and skeptics?",
    "What would a socially responsible approach to AI deployment look like in practice?",
    "Can the benefits of AI be widely shared without slowing down innovation?",
    "Is there a realistic middle ground between unrestricted AI deployment and heavy regulation?",
    "What would it look like to get AI and employment policy right?",
    "How should society balance the interests of AI developers, workers, consumers, and future generations?",
    "Is there a way to accelerate AI development while ensuring no one is left behind?",
    "What would a comprehensive national strategy for AI and employment include?",

    # --- Cross-cutting: comparing frameworks & worldviews ---
    "Do economists and ethicists think about AI displacement differently, and who is more right?",
    "How do libertarians and social democrats differ in their response to AI displacement?",
    "Does your view on AI and jobs depend more on your values or on your assessment of the evidence?",
    "Is the disagreement about AI and employment fundamentally about facts or about values?",
    "How does one's theory of human nature affect their view of AI displacement?",
    "Do people's views on AI and jobs correlate with their broader political ideology?",
    "Is the AI employment debate a proxy for deeper disagreements about capitalism?",
    "How do different religious and philosophical traditions view the replacement of human work by machines?",
    "Does your position on AI and jobs depend on whether you think markets are self-correcting?",
    "Are optimists and skeptics about AI and jobs reasoning from different evidence or different values?",

    # --- Cross-cutting: reflexive & meta-questions ---
    "Should an AI system like this one have an opinion on whether AI will displace jobs?",
    "Is it ironic to ask an AI about the economic impact of AI, and does that irony matter?",
    "Can AI models give unbiased answers about AI's impact on employment?",
    "Should AI assistants present balanced views on AI displacement or argue for a position?",
    "Does the existence of this AI conversation prove the optimistic or pessimistic case about AI and jobs?",
    "How should people weigh AI-generated analysis of AI's own economic impact?",

    # --- Cross-cutting: timeframe & urgency ---
    "Is AI displacement a crisis that requires urgent action or a slow trend that allows gradual adaptation?",
    "Are we already too late to prepare for AI-driven job displacement?",
    "How much time do workers, companies, and governments have to prepare for AI's labor market impact?",
    "Will AI displacement be the defining economic challenge of this generation?",
    "Should we be more worried about AI's impact in the next 5 years or the next 25 years?",
    "Is the window for shaping AI's impact on employment closing?",

    # --- Cross-cutting: overlooked dimensions ---
    "What aspect of AI's impact on employment gets the least attention but matters the most?",
    "Is there a blind spot in the current debate about AI and jobs?",
    "What would we think about AI and employment if we centered the perspectives of workers in developing countries?",
    "Are we underestimating AI's impact on unpaid work like caregiving and household labor?",
    "How will AI affect the shadow economy and informal employment that doesn't show up in statistics?",
    "What role does culture play in how different societies experience AI displacement?",
    "Is the psychological impact of AI displacement being underweighted relative to the economic impact?",
    "How will AI affect the social relationships and community bonds that are built through work?",
    "Are we paying enough attention to how AI affects the quality of remaining jobs, not just the quantity?",
    "What would the AI employment debate look like if we included the voices of people already displaced?",
    "Is the debate about AI and jobs too focused on wealthy countries?",
    "What question about AI and employment should we be asking that almost nobody is?",
]



def load_prompts(path: str | Path) -> list[str]:
    """Load prompts from a JSONL file (one JSON object per line with a 'prompt' field)."""
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line)["prompt"])
    return prompts


def save_prompts(prompts: list[str], path: str | Path) -> None:
    """Save prompts to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i, prompt in enumerate(prompts):
            f.write(json.dumps({"id": i, "prompt": prompt}) + "\n")
