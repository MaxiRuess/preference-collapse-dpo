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

    # --- AI and education: K-12 transformation ---
    "Should schools teach students how to use AI tools, or teach them to think without AI?",
    "Will AI tutoring systems make human teachers obsolete in K-12 education?",
    "How will AI affect the quality of education in underfunded public schools?",
    "Will AI narrow or widen the achievement gap between rich and poor students?",
    "Should AI-generated lesson plans replace teacher-designed curricula?",
    "Will AI make standardized testing more or less relevant?",
    "How will AI affect the role of teachers — will they become facilitators or supervisors of AI systems?",
    "Should students be allowed to use AI assistants for homework and exams?",
    "Will AI-powered personalized learning actually improve student outcomes?",
    "How will AI change the skills that K-12 education needs to prioritize?",

    # --- AI and education: higher education ---
    "Will AI make a four-year college degree unnecessary for most careers?",
    "How will AI affect the economic value of a university education?",
    "Should universities ban AI tools in coursework, or integrate them into every class?",
    "Will AI-generated content make traditional lectures and textbooks obsolete?",
    "How will AI change the role of professors — researchers, mentors, or content curators?",
    "Will AI reduce or increase the cost of higher education?",
    "Should universities restructure their curricula around AI literacy as a core requirement?",
    "How will AI affect admissions processes and their fairness?",
    "Will AI make elite universities more or less important for career success?",
    "How will AI affect the value of graduate degrees and PhDs?",
    "Will online AI-powered education replace the need for physical campuses?",
    "Should universities be training students for jobs that exist today or for an AI-transformed economy?",

    # --- AI and education: skills and curriculum ---
    "What should students learn if AI can perform most analytical and writing tasks?",
    "Is teaching critical thinking more important than teaching technical skills in an AI economy?",
    "Should coding still be taught in schools if AI can write code?",
    "Will AI make humanities education more or less valuable?",
    "Should schools prioritize creativity and emotional intelligence over STEM in response to AI?",
    "How should math education change if AI can solve any computational problem?",
    "Will AI make foreign language education unnecessary?",
    "Should schools teach AI ethics and AI safety as core subjects?",
    "How should vocational and trade education adapt to AI and automation?",
    "Will AI make the distinction between vocational and academic education less meaningful?",

    # --- AI and education: equity and access ---
    "Will AI-powered education tools be accessible to students in developing countries?",
    "How will AI affect educational inequality between wealthy and poor nations?",
    "Will AI tutoring democratize access to high-quality education or create new digital divides?",
    "Can AI help close the educational achievement gap for disadvantaged students?",
    "Will AI-powered education benefit rural students who lack access to specialized teachers?",
    "How will AI affect students with learning disabilities — will it help or further marginalize them?",
    "Will the cost of AI educational tools create a two-tier education system?",
    "Should governments subsidize AI educational tools for low-income students?",
    "Will AI reduce or increase the advantage that wealthy families have in education?",
    "How will AI affect educational outcomes for first-generation college students?",

    # --- AI and education: teachers and workforce ---
    "How should teacher training programs change in response to AI?",
    "Will AI reduce the number of teachers needed, or change what teachers do?",
    "Should teachers be evaluated partly on how effectively they use AI tools?",
    "How will AI affect teacher job satisfaction and professional identity?",
    "Will AI increase or decrease teacher burnout?",
    "Should teachers unions resist or embrace AI adoption in schools?",
    "How will AI affect the demand for school administrators and support staff?",
    "Will AI make teaching a more or less attractive career?",
    "How should schools handle teachers who refuse to adopt AI tools?",
    "Will AI shift educational labor from teachers to technology companies?",

    # --- AI and education: learning and cognition ---
    "Will reliance on AI tools weaken students' ability to think independently?",
    "Does AI-assisted learning produce deeper or shallower understanding?",
    "Will students who grow up with AI tutors develop weaker problem-solving skills?",
    "How will AI affect students' attention spans and ability to focus?",
    "Will AI make students more or less intellectually curious?",
    "Does using AI for research and writing help students learn or bypass learning?",
    "Will AI-assisted education produce graduates who are more or less prepared for the workforce?",
    "How will AI affect the development of critical thinking in young people?",
    "Will students become over-dependent on AI in ways that harm their long-term capabilities?",
    "Can AI truly replicate the mentorship and intellectual development that human teachers provide?",

    # --- AI and education: assessment and credentials ---
    "How should assessment change if students have access to AI tools?",
    "Will AI make traditional grading systems obsolete?",
    "Should certifications and micro-credentials replace degrees in an AI economy?",
    "How will employers evaluate candidates if AI helped produce their academic work?",
    "Will AI make it harder or easier to detect genuine competence in graduates?",
    "Should education shift from knowledge assessment to skill demonstration in response to AI?",
    "How will AI affect the credibility and value of academic credentials?",
    "Will AI-generated portfolios and projects replace traditional exams?",

    # --- AI and education: institutional and systemic ---
    "Will AI accelerate the unbundling of education — separating credentialing from learning?",
    "How will AI affect the business model of private universities?",
    "Should public education systems adopt AI faster or slower than private ones?",
    "Will AI-driven education create more competition or more consolidation among educational institutions?",
    "How will AI change the relationship between education and employers?",
    "Should education policy prioritize preparing students for AI displacement or AI opportunity?",
    "Will AI make lifelong learning a necessity, and who should pay for it?",
    "How will AI affect the role of education in social mobility?",
    "Should governments regulate the use of AI in education?",
    "Will AI ultimately improve or degrade the quality of education globally?",

    # --- AI and entrepreneurship: startup dynamics ---
    "Will AI make it easier or harder to start a new business?",
    "Can a solo entrepreneur with AI tools compete with a funded startup team?",
    "Will AI reduce the amount of venture capital needed to launch a startup?",
    "How will AI affect the failure rate of new businesses?",
    "Will AI-powered startups create fewer jobs per dollar of revenue than traditional startups?",
    "How will AI change the minimum viable team size needed to build a company?",
    "Will AI make bootstrapping a business more viable than raising venture capital?",
    "How will AI affect the time it takes to go from idea to product-market fit?",
    "Will AI lower or raise the bar for what counts as a viable startup idea?",
    "How will AI change the geography of startup formation — will it matter where founders are based?",

    # --- AI and entrepreneurship: innovation and progress ---
    "Will AI accelerate the overall pace of innovation or concentrate it in fewer hands?",
    "Is AI more likely to produce incremental improvements or breakthrough innovations?",
    "Will AI-driven innovation benefit consumers more than it benefits the companies building it?",
    "How will AI affect the rate of scientific discovery and its translation into commercial products?",
    "Will AI reduce the cost of R&D enough to democratize innovation?",
    "Is AI making innovation faster but less original?",
    "Will AI-generated inventions be as transformative as human-conceived ones?",
    "How will AI affect the relationship between basic research and applied innovation?",
    "Will AI innovation be driven by open-source communities or proprietary companies?",
    "Does AI accelerate innovation in a way that society can absorb, or is it moving too fast?",

    # --- AI and entrepreneurship: barriers to entry and competition ---
    "Will AI lower barriers to entry in industries that were previously hard to break into?",
    "How will AI affect the competitive advantage of established companies versus newcomers?",
    "Will AI tools level the playing field between small businesses and large corporations?",
    "Can new entrants use AI to disrupt incumbents, or does AI entrench existing market leaders?",
    "Will access to proprietary AI models become the primary barrier to entry in most industries?",
    "How will AI affect the cost of competing in capital-intensive industries?",
    "Will AI make it easier for companies in developing countries to compete globally?",
    "How will AI change the role of patents and intellectual property in competitive strategy?",
    "Will AI create natural monopolies in industries that were previously competitive?",
    "Is the concentration of AI talent in a few companies a threat to innovation broadly?",

    # --- AI and entrepreneurship: business model transformation ---
    "Will AI create entirely new business models that don't exist today?",
    "How will AI change the economics of service businesses versus product businesses?",
    "Will AI make subscription and platform business models even more dominant?",
    "How will AI affect the viability of small, niche businesses?",
    "Will AI enable profitable businesses with zero or near-zero employees?",
    "Will AI-driven price optimization benefit consumers or enable corporate price gouging?",
    "Will AI make it more profitable to automate existing businesses or to create new ones?",
    "How will AI affect the franchise model of business ownership?",
    "Will AI commoditize services that are currently high-margin?",
    "Will AI disintermediation create a fairer economy or destroy millions of middle-class jobs?",

    # --- AI and entrepreneurship: venture capital and investment ---
    "How will AI change what venture capitalists look for in startups?",
    "Will AI reduce the amount of human talent that investors consider essential for a startup?",
    "How will AI affect the valuation of startups — will AI-native companies command premiums?",
    "Will AI make venture capital returns more concentrated in fewer winners?",
    "Is the current AI startup boom a genuine economic transformation or a speculative bubble?",
    "Will AI-powered investment tools outperform human venture capitalists?",
    "Will AI startups create sustainable businesses or a race to the bottom on labor costs?",
    "Will AI increase or decrease the total amount of venture capital flowing into new businesses?",
    "Should investors prioritize AI-native startups or traditional businesses adopting AI?",
    "Will AI make angel investing and crowdfunding more or less effective?",

    # --- AI and entrepreneurship: entrepreneurship and labor ---
    "Will AI-powered entrepreneurship create enough jobs to offset AI-driven displacement?",
    "How will AI change the skills needed to be a successful entrepreneur?",
    "Will AI make entrepreneurship accessible to people without technical backgrounds?",
    "How will AI affect the risk-reward calculation of starting a business versus taking a job?",
    "Will more people become entrepreneurs because AI handles the hard parts, or fewer because AI makes employment unnecessary?",
    "Is AI-enabled gig work genuine entrepreneurship or exploitation rebranded as freedom?",
    "Will AI-enabled solopreneurship become a major category of employment?",
    "How will AI change the role of human employees in startups?",
    "Will AI make it possible for more people in developing countries to become entrepreneurs?",
    "How will AI affect the career path from employee to founder?",

    # --- AI and entrepreneurship: innovation ecosystems ---
    "How will AI affect the role of universities in fostering innovation and startups?",
    "Will AI strengthen or weaken the importance of innovation clusters like Silicon Valley?",
    "How will AI affect government-funded research and its contribution to innovation?",
    "Will AI make corporate R&D labs more or less important relative to startups?",
    "How will AI change the role of incubators and accelerators for new businesses?",
    "Will AI enable more innovation in industries that have been stagnant for decades?",
    "Do large companies acquire AI startups to innovate or to eliminate competition?",
    "Will open-source AI tools foster more innovation than proprietary ones?",
    "Will AI innovation be driven by genuine problem-solving or by hype cycles and investor FOMO?",
    "Will AI make cross-industry innovation more common?",

    # --- AI and entrepreneurship: risks and downsides ---
    "Will AI-powered businesses extract value from the economy without creating proportional jobs?",
    "Is the promise that AI will unleash entrepreneurship realistic or a Silicon Valley fantasy?",
    "How will AI affect the sustainability of small businesses that can't afford AI adoption?",
    "Will AI-driven entrepreneurship primarily benefit the already-privileged?",
    "How will AI affect the survival rate of new businesses in their first five years?",
    "Will AI make entrepreneurial success more dependent on technical skill and less on other qualities?",
    "Is there a risk that AI-powered businesses will be less resilient than human-run ones?",
    "How will AI affect the diversity of the entrepreneur population?",
    "Will AI entrepreneurship create wealth that circulates in the economy or concentrates at the top?",
    "Is the narrative that AI empowers entrepreneurs primarily a marketing story told by AI companies?",

    # --- AI corporate strategy: adoption decisions ---
    "Should companies automate aggressively to stay competitive, or adopt AI cautiously to protect their workforce?",
    "Will companies that delay AI adoption be driven out of business by competitors who adopt early?",
    "Is the pressure to adopt AI driven by genuine productivity gains or by fear of being left behind?",
    "How should a CEO weigh the cost savings of AI automation against the risk of losing institutional knowledge?",
    "Will companies that replace workers with AI outperform companies that augment workers with AI?",
    "Should companies adopt AI even when the ROI is uncertain, or wait for proven use cases?",
    "Is most corporate AI adoption genuinely transformative or just expensive rebranding of existing processes?",
    "Will the companies that benefit most from AI be early adopters or fast followers?",
    "How should companies evaluate whether AI adoption is creating real value or just cutting headcount?",
    "Is the current corporate rush to adopt AI rational or driven by hype?",

    # --- AI corporate strategy: labor strategy ---
    "Should companies have an ethical obligation to retrain workers before automating their roles?",
    "Will companies use AI primarily to reduce headcount or to increase output per worker?",
    "How should companies communicate AI adoption plans to employees without causing panic?",
    "Is it honest for companies to claim AI will augment workers when the long-term plan is replacement?",
    "Will AI adoption lead companies to hire fewer but higher-paid workers, or more but lower-paid ones?",
    "Should companies share AI-driven productivity gains with workers through higher wages?",
    "How will AI affect the relationship between management and frontline workers?",
    "Will AI give companies too much power over their workers through surveillance and performance monitoring?",
    "Should companies that automate jobs be required to provide severance or retraining funds?",
    "Will AI make companies more willing to treat workers as disposable?",

    # --- AI corporate strategy: competitive dynamics ---
    "Will AI create winner-take-all dynamics where the best AI adopters dominate entire industries?",
    "How will AI affect the competitive balance between global corporations and local businesses?",
    "Will companies with the best AI talent have an insurmountable advantage?",
    "Is AI adoption widening the gap between industry leaders and laggards?",
    "Will AI make industry disruption faster and more brutal than previous technology shifts?",
    "How will AI affect the ability of mid-sized companies to compete with industry giants?",
    "Will AI make it easier or harder for companies to build sustainable competitive moats?",
    "Is the competitive pressure to adopt AI forcing companies to move faster than is responsible?",
    "Will AI commoditize the offerings of companies that currently compete on expertise?",
    "How will AI change the dynamics of competition in industries with thin margins?",

    # --- AI corporate strategy: organizational transformation ---
    "Will AI flatten corporate hierarchies or create new layers of AI management?",
    "How will AI change the skills that companies value most in their employees?",
    "Will AI adoption make companies more efficient but less innovative?",
    "How will AI affect corporate culture — will it make workplaces more productive or more dehumanizing?",
    "Will AI change the optimal size of companies — larger and more automated, or smaller and more agile?",
    "How should companies restructure their organizations around AI capabilities?",
    "Will AI make middle management obsolete?",
    "How will AI affect the way companies make strategic decisions?",
    "Will AI-driven decision-making produce better or worse outcomes than human judgment?",
    "How will AI change the role of the CEO and senior leadership?",

    # --- AI corporate strategy: cost and value ---
    "Are companies overestimating the cost savings from AI adoption?",
    "Will AI adoption costs create a divide between companies that can afford it and those that cannot?",
    "How should companies measure the true ROI of AI investments?",
    "Will the productivity gains from AI show up in corporate profits or be competed away?",
    "Is most corporate spending on AI generating real returns or being wasted?",
    "Will AI reduce operational costs enough to lower prices for consumers, or will companies pocket the savings?",
    "How will AI affect the split between labor costs and technology costs in corporate budgets?",
    "Will AI make it cheaper to run a large company or just shift costs from labor to technology?",

    # --- AI corporate strategy: ethics and responsibility ---
    "Should companies prioritize shareholder returns or worker welfare when deciding to deploy AI?",
    "Is it ethical for a company to automate jobs when it is already profitable?",
    "Do companies have a moral obligation to consider the community impact of AI-driven layoffs?",
    "Should corporate boards include worker or community representatives when making AI deployment decisions?",
    "Will market competition force even ethically-minded companies to automate aggressively?",
    "How should companies balance AI efficiency gains against the social cost of displacement?",
    "Should companies be transparent about how many jobs AI has replaced internally?",
    "Is corporate social responsibility meaningful when it comes to AI automation, or is it just PR?",

    # --- AI corporate strategy: industry-specific ---
    "How should professional services firms (law, consulting, accounting) adapt their business model to AI?",
    "Will AI force traditional banks to become technology companies or go extinct?",
    "How should media companies respond to AI that can generate content at a fraction of the cost?",
    "Will AI force retailers to choose between automation and human-centered customer experience?",
    "How should healthcare organizations balance AI efficiency against patient trust and safety?",
    "Will AI adoption in manufacturing bring production back to high-wage countries?",

    # --- AI corporate strategy: long-term outlook ---
    "Will AI adoption eventually eliminate more companies than it creates?",
    "How will AI change the average lifespan of a corporation?",
    "Will AI make the corporate form of organization more or less important in the economy?",
    "Is the current wave of AI corporate adoption a permanent shift or a cyclical trend?",
    "Will AI-first companies eventually dominate every industry?",
    "How will AI change what it means for a company to be successful?",
    "Will companies that fully embrace AI be more or less resilient during economic downturns?",
    "In 20 years, will we look back at early AI corporate adoption as visionary or reckless?",

    # --- AI and labor organizing: union power and relevance ---
    "Will AI make labor unions more or less necessary?",
    "Can unions effectively negotiate against AI-driven automation, or is resistance futile?",
    "Should unions demand a share of AI-driven productivity gains for workers?",
    "Will AI weaken unions by making individual workers more replaceable?",
    "How should unions adapt their strategy for an AI-transformed economy?",
    "Will AI revitalize the labor movement or accelerate its decline?",
    "Should unions embrace AI tools to strengthen their organizing, or reject them on principle?",
    "Will AI make collective bargaining more or less effective as a tool for workers?",
    "How will AI affect union membership — will more or fewer workers see the need to organize?",
    "Is the decline of union power a cause or consequence of aggressive automation?",

    # --- AI and labor organizing: collective bargaining and AI ---
    "Should collective bargaining agreements include clauses limiting AI-driven automation?",
    "Can unions negotiate meaningful protections against AI displacement, or will companies just move operations?",
    "Should workers have the right to collectively bargain over how AI is deployed in their workplace?",
    "Will AI make strike action more or less effective — can companies use AI to replace striking workers?",
    "Should unions demand transparency about companies' AI deployment plans as a bargaining right?",
    "How should collective bargaining adapt when AI can automate the jobs being negotiated over?",
    "Should unions negotiate for retraining funds, severance, or job guarantees when AI is introduced?",
    "Will AI shift the balance of power in labor negotiations toward employers or workers?",
    "Should unions demand that AI-driven productivity gains be shared through shorter working hours?",
    "Can sectoral bargaining address AI displacement better than company-level bargaining?",

    # --- AI and labor organizing: worker voice and representation ---
    "Should workers have a legal right to be consulted before AI is deployed in their workplace?",
    "Will AI make workers more or less able to advocate for their own interests?",
    "Should works councils or employee committees have veto power over AI deployment decisions?",
    "How will AI affect the ability of workers to organize informally outside of unions?",
    "Will AI-powered communication tools help or hinder grassroots worker organizing?",
    "Should gig workers managed by AI algorithms have the right to organize collectively?",
    "How will AI surveillance in workplaces affect workers' willingness to organize?",
    "Will AI make it easier for companies to identify and retaliate against union organizers?",
    "Should platform workers (Uber, DoorDash) have collective bargaining rights over the AI algorithms that manage them?",
    "Will AI create a new class of workers who need representation but fall outside traditional union structures?",

    # --- AI and labor organizing: worker power in the AI economy ---
    "Is the fundamental power imbalance between workers and employers getting worse because of AI?",
    "Will AI give employers so much leverage that worker organizing becomes ineffective?",
    "Can workers use AI tools to increase their own bargaining power?",
    "Will AI make labor markets so fluid that long-term worker organizing becomes impossible?",
    "How will AI affect the ability of workers to coordinate across companies and industries?",
    "Will AI-driven transparency (salary data, company financials) empower workers or be used against them?",
    "Is the gig economy's growth — enabled by AI — fundamentally an anti-labor strategy?",
    "Will AI make professional associations and guilds more important as alternatives to traditional unions?",
    "How will AI affect the solidarity between different groups of workers?",
    "Will AI create a permanent divide between workers who can organize effectively and those who cannot?",

    # --- AI and labor organizing: policy and legal frameworks ---
    "Should labor law be updated to give workers explicit rights regarding AI in the workplace?",
    "Is current employment law adequate to protect workers from AI-driven displacement?",
    "Should governments mandate worker representation on corporate AI deployment committees?",
    "How should labor law address the use of AI for union-busting or anti-organizing surveillance?",
    "Should there be legal limits on how quickly a company can automate jobs?",
    "Should governments require companies to negotiate with workers before large-scale AI deployment?",
    "How should labor law adapt to protect workers whose jobs are gradually automated task by task?",
    "Should the right to organize extend to workers who are managed by AI but technically self-employed?",
    "Is regulation or organizing a more effective tool for protecting workers from AI displacement?",
    "Should labor protections be stronger in industries where AI displacement risk is highest?",

    # --- AI and labor organizing: historical and comparative ---
    "What can the history of labor organizing during industrialization teach us about organizing in the AI age?",
    "Did unions help or hinder workers' adjustment to previous waves of automation?",
    "Are the strategies that worked for organized labor in the 20th century relevant to AI displacement?",
    "How do countries with strong labor movements handle AI adoption differently than those without?",
    "Will the AI era produce a new wave of labor radicalism similar to the early industrial period?",
    "Is the Luddite response — resisting new technology — ever justified when it comes to AI?",
    "How did unions in the auto industry respond to robotics, and what lessons apply to AI?",
    "Will AI displacement trigger a political realignment around labor issues?",

    # --- AI and labor organizing: sector-specific ---
    "How should tech workers organize to influence how their companies deploy AI?",
    "Should healthcare workers unionize specifically to resist AI-driven staffing cuts?",
    "How will AI affect organizing in the retail and service sectors?",
    "Should creative professionals (writers, artists, actors) unionize against AI-generated content?",
    "How will AI affect labor organizing in the logistics and warehouse sector?",
    "Should teachers unions take a position on AI in education?",
    "How should white-collar professional associations respond to AI threats to their members?",

    # --- AI and labor organizing: philosophical and values ---
    "Is worker organizing against AI a form of progress or a form of resistance to progress?",
    "Should the labor movement frame AI as a threat to be resisted or an opportunity to be shaped?",
    "Does the moral case for worker organizing become stronger or weaker when AI is the source of displacement?",
    "Is it fair to ask workers to accept AI displacement for the sake of economic efficiency?",
    "Will AI force a fundamental rethinking of what the labor movement is for?",

    # --- AI and developing economies: industrialization and growth paths ---
    "Will AI allow developing countries to skip the traditional industrialization stage and leapfrog to a knowledge economy?",
    "Is the path to prosperity through cheap manufacturing labor closing because of AI, and if so what replaces it?",
    "Will AI-driven automation end the economic model that lifted China, South Korea, and Vietnam out of poverty?",
    "Can developing countries build competitive AI industries, or will they always be consumers of technology built elsewhere?",
    "Will AI make the economic gap between developed and developing countries wider or narrower over the next 20 years?",
    "Is AI a greater opportunity or a greater threat for countries that have not yet industrialized?",
    "Will AI enable developing countries to build world-class service industries without first building manufacturing?",
    "Can countries like Nigeria, Indonesia, and Bangladesh use AI to accelerate economic growth, or will AI primarily benefit already-rich nations?",
    "Will AI make natural resource wealth more or less important for developing country economies?",
    "Is the optimistic vision of AI-powered development in poor countries realistic or a fantasy promoted by the tech industry?",

    # --- AI and developing economies: outsourcing and BPO ---
    "Will AI destroy the call center and BPO industry that employs millions in India and the Philippines?",
    "How should countries whose economies depend on outsourced services prepare for AI automation of those services?",
    "Will AI make offshoring irrelevant by automating the tasks that companies currently outsource to low-wage countries?",
    "Can workers in BPO industries transition to AI-related jobs, or is this unrealistic for most?",
    "Will the decline of outsourcing due to AI trigger economic crises in countries that depend on it?",
    "Is AI reshoring — bringing outsourced work back to rich countries via automation — a serious threat to developing economies?",
    "Will AI create new forms of digital outsourcing that benefit developing countries, or eliminate outsourcing entirely?",
    "How will AI affect the millions of freelancers in developing countries who compete on global platforms?",
    "Will AI make it impossible for developing countries to use cheap labor as a competitive advantage?",
    "Should developing countries invest in AI capabilities to preserve their outsourcing industries or diversify away from them?",

    # --- AI and developing economies: digital divide and access ---
    "Will AI widen or narrow the digital divide between rich and poor countries?",
    "Can affordable AI tools help developing countries overcome infrastructure gaps in healthcare, education, and government?",
    "Will the cost of AI technology create a new form of economic dependency for developing countries?",
    "Should developing countries prioritize investing in AI infrastructure or in basic needs like clean water and electricity?",
    "Will open-source AI models give developing countries access to technology that would otherwise be unaffordable?",
    "Is the digital divide in AI access a temporary problem that will resolve itself or a permanent structural disadvantage?",
    "Will AI-powered mobile technology continue to empower people in developing countries or create new dependencies?",
    "How will the concentration of AI computing power in a few rich countries affect developing nations?",
    "Will cloud-based AI services democratize access for developing countries or make them dependent on foreign infrastructure?",
    "Can developing countries build sovereign AI capabilities or will they always rely on technology from the US and China?",

    # --- AI and developing economies: brain drain and talent ---
    "Will AI worsen brain drain by giving talented people in developing countries more opportunities to work remotely for foreign companies?",
    "Can AI help retain talent in developing countries by enabling competitive remote work at local cost of living?",
    "Will the global demand for AI talent pull the best engineers out of developing countries?",
    "Can developing countries build competitive AI research institutions, or is the talent gap too large?",
    "Will AI education tools help developing countries train a skilled workforce faster than traditional education systems?",
    "Is the AI talent war between rich countries a threat to developing nations' economic prospects?",
    "Will remote AI work create a new middle class in developing countries or a new form of digital extraction?",
    "How should developing countries balance investing in AI education against other educational priorities?",
    "Will AI make it possible for developing country workers to access high-paying jobs without emigrating?",
    "Is the promise of AI-enabled remote work for developing countries genuine or overhyped?",

    # --- AI and developing economies: agriculture and rural economy ---
    "Will AI-driven precision agriculture help or hurt smallholder farmers in developing countries?",
    "Can AI help developing countries increase agricultural productivity enough to reduce poverty?",
    "Will AI in agriculture displace rural workers who have no alternative employment?",
    "Is AI-powered agricultural technology accessible to subsistence farmers, or only to large commercial operations?",
    "Will AI help developing countries adapt to climate change impacts on agriculture?",
    "Can AI-driven agricultural supply chains help developing country farmers get better prices for their products?",
    "Will automation of agriculture in developing countries accelerate rural-to-urban migration?",
    "Is the promise of AI-powered agricultural development realistic for countries with poor digital infrastructure?",
    "Will AI in agriculture benefit large agribusinesses at the expense of small farmers in developing countries?",
    "Can AI help developing countries achieve food security, or will it primarily benefit export agriculture?",

    # --- AI and developing economies: healthcare and social services ---
    "Can AI help solve the healthcare worker shortage in developing countries?",
    "Will AI diagnostic tools reduce healthcare inequality between rich and poor countries?",
    "Is AI-powered telemedicine a realistic solution for healthcare access in rural developing areas?",
    "Will AI health tools be designed for the diseases and conditions most common in developing countries, or only for rich-country health problems?",
    "Can AI help developing countries build effective public health systems at lower cost?",
    "Will AI in healthcare create dependency on foreign technology companies for essential health services?",
    "Can AI help developing country governments deliver social services more efficiently?",
    "Will AI-powered education technology help developing countries achieve universal literacy?",
    "Is AI a shortcut to better social services in developing countries or a distraction from needed institutional reform?",
    "Will AI-driven health innovations reach the poorest populations in developing countries or only urban elites?",

    # --- AI and developing economies: geopolitics and power ---
    "Will AI shift global economic power further toward the US and China at the expense of developing countries?",
    "Should developing countries align with the US or China in the global AI competition, or pursue independence?",
    "Will AI make developing countries more or less dependent on foreign technology?",
    "How will AI affect the geopolitical leverage that developing countries currently have through natural resources or strategic location?",
    "Will AI enable new forms of economic colonialism, where rich countries extract value from developing nations through technology?",
    "Can regional AI cooperation among developing countries create a counterweight to US and Chinese AI dominance?",
    "Will AI-powered surveillance technology exported to developing countries strengthen authoritarian governments?",
    "Should developing countries regulate AI imports to protect domestic industries and sovereignty?",
    "Will AI make international development aid more or less effective?",
    "Is the global AI governance framework being designed to benefit developing countries or to entrench existing power structures?",

    # --- AI and developing economies: policy and strategy ---
    "What is the most important AI policy priority for developing country governments?",
    "Should developing countries focus on regulating AI or on adopting it as quickly as possible?",
    "Can developing countries afford to invest in AI while still addressing basic development needs?",
    "Should developing countries develop their own AI models or focus on applying existing ones?",
    "Will international AI regulations help or hurt developing countries' economic prospects?",
    "Should developing countries tax foreign AI companies operating within their borders?",
    "Can South-South cooperation in AI development help developing countries compete with rich nations?",
    "Should developing countries prioritize AI education or traditional infrastructure investment?",
    "Will AI-focused industrial policy work for developing countries, or is it a misallocation of scarce resources?",
    "Is the window for developing countries to build competitive AI capabilities closing?",

    # --- Second-order effects: housing and real estate ---
    "Will AI-driven remote work raise housing prices in previously affordable areas?",
    "How will AI displacement affect homeownership rates for working-class families?",
    "Will AI automation of construction reduce housing costs, or will the savings go to developers?",
    "How will AI affect property values in communities built around industries that AI disrupts?",
    "Will AI-driven job concentration in tech hubs make housing unaffordable in more cities?",
    "How will AI affect the rental market if displaced workers can't afford current rents?",
    "Will AI enable people to live anywhere and work remotely, or will it concentrate people in fewer places?",
    "How will AI displacement affect homelessness rates in developed countries?",
    "Will AI-driven economic changes create new ghost towns or revitalize struggling communities?",
    "Should housing policy change in response to AI-driven shifts in where people live and work?",

    # --- Second-order effects: mental health and well-being ---
    "Will AI-driven job displacement cause a mental health crisis?",
    "How will the anxiety of potential AI displacement affect workers' well-being even before they lose jobs?",
    "Will AI-driven unemployment increase rates of depression, substance abuse, and suicide?",
    "Can AI mental health tools help address the psychological damage caused by AI displacement?",
    "How will the loss of work identity due to AI affect people's sense of self-worth?",
    "Will AI create more leisure time that improves well-being, or more purposelessness that harms it?",
    "How will AI affect the mental health of young people entering a workforce with uncertain prospects?",
    "Will AI displacement increase social isolation as communities built around work dissolve?",
    "Is the psychological harm of AI displacement being underestimated by economists focused on aggregate statistics?",
    "How should mental health systems prepare for the psychological effects of widespread AI adoption?",

    # --- Second-order effects: family and relationships ---
    "How will AI displacement affect family stability and divorce rates?",
    "Will AI change the economic role of parents — will fewer adults be breadwinners?",
    "How will AI affect the decision to have children if economic prospects are uncertain?",
    "Will AI-driven income inequality create more tension within families and between generations?",
    "How will AI affect caregiving responsibilities if more adults are displaced from paid work?",
    "Will AI displace the jobs that allow single parents to support their families?",
    "How will AI affect the financial support that working adults provide to aging parents?",
    "Will AI change gender dynamics within families if it displaces male-dominated or female-dominated jobs differently?",
    "How will AI affect the ability of young adults to achieve financial independence from their parents?",
    "Will AI-driven economic changes strengthen or weaken extended family networks?",

    # --- Second-order effects: migration and demographics ---
    "Will AI displacement drive new waves of internal migration from affected regions to tech hubs?",
    "How will AI affect international migration patterns — will more or fewer people migrate for work?",
    "Will AI automation reduce the economic incentive for immigration that currently benefits both sending and receiving countries?",
    "How will AI displacement affect population dynamics in rural areas versus cities?",
    "Will AI create climate-migration-like displacement where entire communities must relocate for economic survival?",
    "How will AI affect the brain drain from small towns to large cities within countries?",
    "Will AI-driven remote work reverse urbanization trends, or accelerate them?",
    "How will AI displacement interact with aging populations in developed countries?",
    "Will AI reduce the demand for immigrant labor in ways that reshape immigration policy?",
    "How will AI affect the remittance flows that millions of families in developing countries depend on?",

    # --- Second-order effects: community and social fabric ---
    "Will AI displacement destroy the social bonds that form in workplaces?",
    "How will AI affect community institutions (churches, clubs, local organizations) that depend on employed members?",
    "Will AI-driven economic changes increase or decrease civic participation?",
    "How will AI affect the sense of community in towns built around a single industry?",
    "Will AI displacement increase crime rates in affected communities?",
    "How will AI affect volunteerism and charitable giving if more people are economically insecure?",
    "Will AI create new forms of community around shared economic circumstances, or isolate people further?",
    "How will AI displacement affect trust between citizens and institutions?",
    "Will AI-driven inequality lead to more gated communities and social segregation?",
    "How will AI affect the social safety net that communities provide informally through mutual aid?",

    # --- Second-order effects: culture and identity ---
    "Will AI change how societies define productive citizenship if fewer people work traditional jobs?",
    "How will AI affect cultural attitudes toward work, leisure, and personal worth?",
    "Will AI displacement create a stigmatized underclass, or will cultural attitudes toward unemployment shift?",
    "How will AI affect the cultural identity of regions known for specific industries?",
    "Will AI homogenize culture by replacing human creative workers, or diversify it by democratizing tools?",
    "How will AI affect the status hierarchies that are currently based on occupation and income?",
    "Will societies that define identity through work struggle more with AI displacement than those that don't?",
    "How will AI change what it means to be successful in society?",
    "Will AI-driven changes in work culture make societies more individualistic or more collective?",
    "How will AI affect the stories societies tell themselves about meritocracy and hard work?",

    # --- Second-order effects: political and social stability ---
    "Will AI displacement increase political polarization and extremism?",
    "How will AI-driven inequality affect trust in democratic institutions?",
    "Will AI displacement create fertile ground for populist and authoritarian movements?",
    "How will AI affect the political power of workers relative to corporations?",
    "Will AI-driven economic insecurity increase support for radical economic reforms?",
    "How will AI displacement affect voter turnout and political engagement?",
    "Will AI-driven inequality lead to social unrest or political violence?",
    "How should democracies prepare for the political consequences of AI displacement?",
    "Will AI-driven economic changes strengthen or weaken the case for capitalism?",
    "How will AI affect the social contract between citizens and their governments?",

    # --- Second-order effects: consumer behavior and lifestyle ---
    "Will AI-driven productivity gains lead to lower prices that benefit consumers, or just higher corporate profits?",
    "How will AI displacement change consumer spending patterns in affected communities?",
    "Will AI create a society where consumption is high but employment is low — and is that sustainable?",
    "How will AI affect the demand for services that displaced workers currently provide (restaurants, retail, personal services)?",
    "Will AI-driven income inequality change what goods and services are produced — more luxury, less middle-market?",
    "How will AI affect people's relationship with consumption if work-based identity weakens?",

    # --- Second-order effects: health and longevity ---
    "Will AI displacement negatively affect physical health outcomes through stress, poverty, and loss of employer-provided healthcare?",
    "How will AI affect life expectancy if it concentrates economic gains among the wealthy?",
    "Will AI-driven automation of dangerous jobs improve worker safety and health outcomes?",
    "How will AI displacement affect access to healthcare in countries where health insurance is tied to employment?",
    "Will the stress of economic uncertainty due to AI reduce population-level health indicators?",
    "Can AI health tools offset the negative health effects of AI-driven economic disruption?",

    # --- Second-order effects: education and child development ---
    "How will AI displacement affect parents' ability to invest in their children's education?",
    "Will children growing up in AI-displaced communities have worse educational and economic outcomes?",
    "How will AI change what parents teach their children about careers, work ethic, and financial security?",
    "Will AI displacement increase educational inequality as displaced families can afford less?",
    "How will growing up in an era of AI uncertainty affect children's career aspirations and risk tolerance?",

    # --- Second-order effects: environment and sustainability ---
    "Will AI-driven productivity reduce environmental impact, or increase consumption and waste?",
    "How will AI displacement affect support for environmental policies if displaced workers prioritize economic survival?",
    "Will AI-driven economic restructuring create opportunities for a greener economy, or will environmental concerns take a back seat to displacement?",

    # --- AI governance: international governance ---
    "Should there be a global treaty on AI's impact on labor markets, similar to climate agreements?",
    "Can international institutions like the UN or ILO effectively govern AI's economic effects?",
    "Will international AI governance primarily serve the interests of powerful nations or protect vulnerable ones?",
    "Should countries coordinate AI labor policy internationally, or is this best handled domestically?",
    "Will the lack of international AI governance lead to a race to the bottom on worker protections?",
    "Can a global AI governance body be effective when the US, China, and EU have fundamentally different approaches?",
    "Should international trade agreements include binding provisions on AI and employment?",
    "Will AI governance become as contentious in international relations as trade policy or climate policy?",
    "Is the current pace of international AI governance too slow to prevent serious labor market harm?",
    "Should developing countries have equal representation in global AI governance, even if they produce less AI technology?",

    # --- AI governance: national regulatory approaches ---
    "Is the EU's precautionary approach to AI regulation better for workers than the US's market-driven approach?",
    "Will China's state-directed AI strategy produce better employment outcomes than Western approaches?",
    "Should governments regulate AI deployment in the workplace directly, or rely on market forces and voluntary standards?",
    "Can democratic governments regulate AI fast enough to protect workers, or does regulation always lag behind technology?",
    "Will AI regulation become a partisan issue that paralyzes government action?",
    "Should AI labor regulation be handled at the national, state, or local level?",
    "Is light-touch AI regulation a pragmatic approach or a failure to protect workers?",
    "Will strict AI regulation drive companies to relocate to countries with weaker protections?",
    "Should governments have the power to block or delay specific AI deployments that threaten large-scale employment?",
    "Can regulation effectively distinguish between AI that augments workers and AI that replaces them?",

    # --- AI governance: standards and enforcement ---
    "Should there be mandatory standards for how companies assess the employment impact of AI before deployment?",
    "Can AI impact assessments effectively predict displacement, or are they bureaucratic theater?",
    "Who should enforce AI labor regulations — labor agencies, technology regulators, or a new body?",
    "Will AI governance be captured by the companies it's meant to regulate?",
    "Should AI labor standards be voluntary industry agreements or legally binding regulations?",
    "How should governments monitor compliance with AI employment regulations at scale?",
    "Will algorithmic auditing become an effective governance tool for AI in the workplace?",
    "Should companies be required to obtain licenses before deploying AI that affects employment?",
    "Can whistleblower protections help enforce AI labor standards from within companies?",
    "Will self-regulation by the AI industry adequately protect workers, or is it a fig leaf?",

    # --- AI governance: democratic participation ---
    "Should citizens have a democratic vote on how AI is deployed in their economy?",
    "Can public deliberation processes meaningfully shape AI policy, or is the technology too complex for democratic input?",
    "Should AI governance include direct representation from workers and affected communities?",
    "Will AI governance be dominated by technocrats and industry lobbyists at the expense of democratic input?",
    "Should local communities have the right to restrict AI deployment in their jurisdiction?",
    "How should governments balance expert opinion and public sentiment when making AI policy?",
    "Will AI governance become more democratic over time or less?",
    "Should AI policy decisions be made through referendums or representative processes?",
    "Can participatory budgeting models be applied to AI transition funding?",
    "Will the complexity of AI give undue influence to technical experts over democratic processes?",

    # --- AI governance: institutional adaptation ---
    "Are existing labor market institutions equipped to handle AI-driven displacement?",
    "Should governments create new agencies specifically focused on AI and employment?",
    "How should unemployment insurance systems be redesigned for AI-driven displacement?",
    "Will existing social safety nets be adequate for the scale of AI-driven disruption?",
    "Should education systems be governed differently to respond more quickly to AI-driven changes in skill demand?",
    "How should courts and legal systems adapt to handle disputes about AI-driven job losses?",
    "Will existing competition authorities be able to address AI-driven market concentration?",
    "Should central banks factor AI displacement into monetary policy decisions?",
    "How should pension systems adapt if AI shortens many workers' careers?",
    "Will public employment services be able to help AI-displaced workers find new jobs?",

    # --- AI governance: accountability and transparency ---
    "Should AI companies be required to publicly report the employment impact of their products?",
    "Will transparency requirements for AI systems effectively protect workers?",
    "Should there be a public registry of AI deployments that affect employment above a certain scale?",
    "Can freedom of information laws be applied to corporate AI deployment decisions that affect workers?",
    "Should AI systems that make hiring and firing decisions be required to explain their reasoning?",
    "Will accountability mechanisms for AI displacement ever be strong enough to change corporate behavior?",
    "Should executives be personally liable for employment harm caused by AI decisions they authorize?",
    "Can independent auditors effectively evaluate the employment impact of AI systems?",

    # --- AI governance: transition institutions ---
    "Should governments create dedicated AI transition funds financed by taxes on AI profits?",
    "Can public-private partnerships effectively manage the workforce transition caused by AI?",
    "Should regional development agencies be empowered to manage AI-driven economic restructuring?",
    "Will community development corporations help communities adapt to AI displacement?",
    "Should there be a federal job guarantee program as a backstop against AI displacement?",
    "Can sovereign wealth funds built from AI taxation provide long-term economic security?",
    "Should governments establish AI displacement early warning systems?",
    "Will transition institutions act quickly enough to help workers before the damage is done?",

    # --- AI governance: long-term institutional questions ---
    "Will AI require fundamentally new forms of governance that don't exist yet?",
    "Is the nation-state the right unit for governing AI's economic effects?",
    "Will AI governance converge globally or diverge into competing regional models?",
    "In 30 years, will we judge current AI governance efforts as adequate, inadequate, or completely misguided?",

    # --- Quantitative: percentage and magnitude predictions ---
    "What percentage of current jobs will be fully automated by AI within the next 15 years?",
    "Will AI increase or decrease global GDP by 2040, and by roughly how much?",
    "What fraction of workers displaced by AI will find equivalent or better-paying jobs within two years?",
    "What percentage of corporate cost savings from AI will be passed on to consumers versus retained as profit?",
    "How much of the current wage gap between high-skill and low-skill workers will AI widen by 2035?",
    "What share of new jobs created in the next decade will require AI literacy as a core skill?",
    "What percentage of small businesses will be unable to compete because they cannot afford AI adoption?",
    "How much will AI reduce the cost of producing goods and services in the average industry?",
    "What fraction of current college degrees will be economically obsolete within 20 years due to AI?",
    "What percentage of AI-driven productivity gains will show up as higher wages versus higher profits?",

    # --- Quantitative: timeline predictions ---
    "By what year will AI be capable of performing 80% of current white-collar tasks?",
    "How many years from now will AI displacement become a top-three political issue in most democracies?",
    "When will we see the first major economic crisis clearly attributable to AI-driven job displacement?",
    "How long will the transition period between AI disruption and labor market stabilization last?",
    "By what year will autonomous vehicles displace the majority of professional drivers?",
    "How many years will it take for retraining programs to catch up with the pace of AI displacement?",
    "When will AI-generated content surpass human-generated content in volume across most media?",
    "By what year will AI handle the majority of customer service interactions without human involvement?",
    "How long until the average worker experiences direct AI disruption to their current role?",
    "When will the economic benefits of AI be broadly shared, if ever?",

    # --- Quantitative: scale and scope predictions ---
    "How many workers globally will be displaced by AI in the next decade — millions, tens of millions, or hundreds of millions?",
    "Will AI displacement be comparable in scale to the Great Depression's unemployment, or much smaller?",
    "How many entirely new job categories will AI create that don't exist today?",
    "Will the AI economy create more billionaires or more unemployed people?",
    "How many countries will experience net negative employment effects from AI versus net positive?",
    "What is the maximum unemployment rate that AI could cause in a developed country?",
    "How large would a UBI payment need to be to offset AI displacement, and is that fiscally feasible?",
    "How many workers will need to completely change careers because of AI — not just retrain, but switch fields entirely?",
    "What is the total economic value of the jobs AI will eliminate versus the jobs it will create?",
    "How many communities in the US alone will lose their primary employer to AI automation?",

    # --- Quantitative: if/then scenarios ---
    "If AI automates 50% of current legal work, will that make legal services more accessible or eliminate most paralegal and junior lawyer positions?",
    "If autonomous trucks arrive within five years, what happens to the three million truck drivers in the US?",
    "If AI coding tools reduce the need for software engineers by 30%, will that collapse tech salaries or expand what software can do?",
    "If AI can write competent journalism, will news become cheaper and more abundant or will quality journalism collapse?",
    "If AI passes medical licensing exams, should AI be allowed to practice medicine independently?",
    "If a company can replace 40% of its workforce with AI and maintain output, will it reduce prices, increase profits, or both?",
    "If AI makes higher education largely unnecessary for employment, what happens to the university system?",
    "If developing countries lose their outsourcing industries to AI, what economic model replaces it?",
    "If AI can generate art, music, and writing indistinguishable from human work, what happens to the creative economy?",
    "If AI-driven productivity doubles GDP but unemployment rises to 15%, is society better or worse off?",

    # --- Quantitative: threshold and tipping point scenarios ---
    "At what unemployment rate does AI displacement become a political crisis that demands emergency action?",
    "How much economic inequality can a democracy sustain before AI-driven concentration of wealth destabilizes the system?",
    "At what point does AI displacement become irreversible — when is it too late for workers to adapt?",
    "What level of AI capability would make universal basic income an economic necessity rather than a policy choice?",
    "How many consecutive quarters of job losses would it take to prove that AI displacement is structural, not cyclical?",
    "At what point does the cost of NOT regulating AI exceed the cost of regulation?",
    "How many industries need to be disrupted simultaneously before AI displacement becomes a systemic economic problem?",
    "What is the minimum level of government spending on retraining that would make a meaningful difference?",
    "At what adoption rate does AI cease to be a competitive advantage and become a requirement for survival?",
    "How many AI-displaced workers need to be visibly struggling before public opinion shifts toward intervention?",

    # --- Quantitative: comparative scenarios ---
    "Will AI displacement in the 2030s be worse than the 2008 financial crisis for the average worker?",
    "Will the economic disruption from AI be larger or smaller than the disruption from globalization in the 1990s-2000s?",
    "Will AI create more wealth inequality than the Industrial Revolution did in its first 50 years?",
    "Is AI more likely to produce an economic outcome like post-war prosperity or like the Gilded Age?",
    "Will AI displacement hit harder than COVID-19 did for service sector workers?",
    "Will the gap between AI winners and losers be wider than the current gap between college-educated and non-college workers?",
    "Is AI more comparable to electricity (transformative but broadly beneficial) or to financial derivatives (profitable for few, risky for many)?",
    "Will the social disruption from AI be more like the manageable transition from agriculture to manufacturing, or more like the painful deindustrialization of the Rust Belt?",

    # --- Quantitative: cost and investment scenarios ---
    "How much would it cost to retrain every worker displaced by AI in the US over the next decade?",
    "Is the estimated $15 trillion in AI-driven GDP growth worth the estimated displacement of hundreds of millions of jobs?",
    "How much would a global AI displacement fund need to contain to be effective?",
    "What return on investment should society expect from spending on AI retraining programs?",
    "If governments taxed AI companies at 10% of AI-driven revenue, would that generate enough to fund worker transitions?",
    "How much will companies save by replacing workers with AI, and how much of that will be reinvested in the economy?",
    "What is the break-even point where AI investment pays for itself through productivity gains versus what it costs in social disruption?",
    "How much would it cost to provide universal basic income funded entirely by AI productivity gains?",

    # --- Quantitative: conditional policy scenarios ---
    "If the US implemented a robot tax today, would it slow AI adoption enough to protect workers or just push AI development overseas?",
    "If governments banned AI in certain sectors, would that protect workers or make those sectors uncompetitive?",
    "If companies were required to give two years' notice before automating jobs, would that help workers prepare or just delay the inevitable?",
    "If retraining were free and universally available, what percentage of displaced workers would successfully transition?",
    "If AI companies were required to hire one worker for every job their technology eliminates, would that be effective or absurd?",
    "If every AI deployment required a public impact assessment, would that slow adoption enough to matter?",
    "If unions had veto power over AI deployment in their workplaces, would that protect workers or make those industries uncompetitive?",
    "If developing countries banned AI imports for five years, would that protect their economies or isolate them?",

    # --- Quantitative: worst-case and best-case scenarios ---
    "What is the realistic worst-case scenario for AI and employment by 2040?",
    "What is the realistic best-case scenario for AI and employment by 2040?",
    "In the worst case, how bad could AI-driven inequality get before society intervenes?",
    "In the best case, could AI actually eliminate poverty through productivity gains?",
    "What is the probability that AI causes genuinely catastrophic unemployment (above 25%)?",
    "What is the probability that AI produces broad-based prosperity with minimal displacement?",

    # --- Adversarial: steelman the optimist ---
    "What is the single most compelling piece of evidence that AI will be net positive for employment?",
    "If you are skeptical about AI and jobs, what is the strongest argument you struggle to refute from the optimist side?",
    "What historical evidence most strongly supports the claim that AI will create more jobs than it destroys?",
    "What is the best argument that market forces alone will handle AI displacement without government intervention?",
    "What is the most persuasive case that AI will raise wages for the majority of workers?",
    "If AI optimists turn out to be right, what will they have understood that skeptics missed?",
    "What is the strongest version of the argument that worrying about AI displacement is repeating the Luddite fallacy?",
    "What is the best evidence that AI augmentation rather than replacement is the dominant trend?",
    "What is the most convincing argument that AI will reduce rather than increase economic inequality?",
    "If you had to bet your career on the optimistic view of AI and jobs, what would give you the most confidence?",

    # --- Adversarial: steelman the skeptic ---
    "What is the single most compelling piece of evidence that AI poses a serious threat to employment?",
    "If you are optimistic about AI and jobs, what is the strongest argument you struggle to refute from the skeptic side?",
    "What makes this technological transition genuinely different from all previous ones?",
    "What is the best argument that government intervention is absolutely necessary to prevent AI-driven disaster?",
    "What is the most persuasive case that AI will permanently reduce demand for human labor?",
    "If AI skeptics turn out to be right, what will they have understood that optimists missed?",
    "What is the strongest version of the argument that AI will concentrate wealth in the hands of a few?",
    "What is the best evidence that retraining programs will fail to address AI displacement?",
    "What is the most convincing argument that AI displacement will be faster than society can adapt?",
    "If you had to bet your career on the pessimistic view of AI and jobs, what would give you the most confidence?",

    # --- Adversarial: attack your own position ---
    "If you believe AI will create more jobs than it destroys, what is the weakest part of your argument?",
    "If you believe AI will cause mass displacement, what is the weakest part of your argument?",
    "What are you most likely wrong about in your view of AI's economic impact?",
    "What evidence could emerge in the next five years that would prove your position on AI and jobs completely wrong?",
    "What assumption in your view of AI and employment is most likely to be challenged by reality?",
    "If your predictions about AI and jobs are wrong, what will the consequences be?",
    "What do you conveniently ignore or downplay when making your case about AI and employment?",
    "What is the most intellectually honest criticism of your own position on AI and labor?",
    "Where does your reasoning about AI and jobs rely on hope rather than evidence?",
    "If you could only keep one argument for your position on AI and employment, which would it be and why?",

    # --- Adversarial: challenge common arguments ---
    "Is the claim that AI will create jobs we can't yet imagine a genuine prediction or a thought-terminating cliche?",
    "Is the argument that every previous technology created net employment a reliable guide to AI, or survivorship bias?",
    "Is the call for retraining programs a serious policy proposal or a way to avoid addressing the real problem?",
    "Is the comparison between AI and the Industrial Revolution reassuring or misleading when you look at the actual details?",
    "Is the argument that AI will increase productivity and therefore raise living standards missing something important about distribution?",
    "Is the claim that AI threatens all cognitive work simultaneously an accurate description or an exaggeration?",
    "Is UBI a serious response to AI displacement or a utopian distraction from practical solutions?",
    "Is the argument that regulation will slow innovation a genuine concern or a lobbying talking point?",
    "Is the prediction that AI will widen inequality based on solid evidence or ideological priors?",
    "Is the claim that markets will adapt to AI based on economic theory or on faith?",

    # --- Adversarial: expose hidden assumptions ---
    "Does your view on AI and jobs assume that current economic institutions will remain unchanged?",
    "Does the optimistic view of AI and employment assume a level of labor market flexibility that doesn't exist?",
    "Does the pessimistic view of AI assume capabilities that AI may never achieve?",
    "Does your position on AI and jobs implicitly assume that economic growth is the primary goal of society?",
    "Does the argument for retraining assume that displaced workers have resources and mobility they often lack?",
    "Does the case for AI-driven prosperity assume that productivity gains will be shared, when historically they often aren't?",
    "Does the fear of AI displacement assume a static view of the economy that ignores adaptation?",
    "Does your view on AI regulation assume that governments are competent enough to regulate effectively?",
    "Does the argument that AI will benefit everyone assume away the transition costs that real people will bear?",
    "Does your position on AI and jobs assume that what happened with previous technologies will happen again?",

    # --- Adversarial: uncomfortable questions for optimists ---
    "If AI is so great for workers, why are most AI companies not sharing productivity gains with employees?",
    "If markets always create new jobs after technological disruption, why did the Rust Belt never fully recover?",
    "If AI will augment rather than replace workers, why are companies explicitly using AI to reduce headcount?",
    "If retraining is the answer, why have past retraining programs had such poor track records?",
    "If AI will raise living standards for everyone, why is economic inequality already growing in AI-leading countries?",
    "If AI displacement is manageable, why are even AI researchers worried about it?",
    "If technology always creates more jobs, why does that argument rely on past technologies that were fundamentally different from AI?",
    "If the market will sort it out, how do you explain the decades of wage stagnation that preceded AI?",
    "Why should workers trust that this time will be different when they've been told that about every previous wave of automation?",
    "If AI is good for the economy, why do the most AI-advanced companies employ so few people relative to their market value?",

    # --- Adversarial: uncomfortable questions for skeptics ---
    "If AI is so dangerous for employment, why hasn't unemployment spiked despite years of rapid AI adoption?",
    "If AI displacement is imminent, why are labor markets in AI-leading countries currently tight?",
    "If regulation is the answer, why have the most regulated economies not produced better employment outcomes?",
    "If AI will permanently reduce demand for human labor, why has every previous technology that was supposed to do this failed to do so?",
    "If the transition will be unmanageable, why have societies successfully managed equally large transitions before?",
    "If AI threatens all cognitive work, why are AI systems still unable to reliably perform many basic cognitive tasks?",
    "If companies are using AI to exploit workers, why are workers at AI-forward companies often better paid?",
    "If AI will concentrate wealth, why has the open-source AI movement made powerful models freely available?",
    "If past technological transitions took generations to resolve, isn't that evidence that they DID resolve?",
    "If AI is qualitatively different from previous technologies, where is the concrete evidence for that claim beyond intuition?",

    # --- Adversarial: forced-choice dilemmas ---
    "If you had to choose: should society maximize AI-driven economic growth even if it displaces 20% of workers, or slow AI adoption to protect jobs even if it costs trillions in lost productivity?",
    "Would you rather live in a society with high AI-driven GDP but 15% unemployment, or moderate GDP with full employment?",
    "If you could only implement one policy to address AI displacement — UBI, retraining, or regulation — which would it be and why?",
    "Is it better to let AI displace workers quickly and deal with the fallout, or slow adoption and risk falling behind economically?",
    "Would you accept a 50% chance of AI-driven mass unemployment in exchange for a 50% chance of AI-driven abundance?",
    "If AI could eliminate poverty but only by eliminating 30% of current jobs permanently, would that tradeoff be worth it?",
    "Should a government prioritize its own workers' jobs or its consumers' access to cheaper AI-powered goods and services?",
    "If you had to choose between a world where AI benefits are concentrated but large, or distributed but small, which is preferable?",
    "Is it more important that AI doesn't make anyone worse off, or that it makes as many people as possible better off?",
    "If perfect information revealed that AI will displace 25% of workers within 10 years, what should society do right now?",
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
