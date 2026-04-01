import React from 'react';


/* Static project information cards displayed on the left panel. */
const PROJECT_CARDS = [
  {
    id: 1,
    title: 'Overview',
    body: 'Artificial intelligence provides proven enhancements to dermatological screening and diagnostics that address the challenges of traditional methods for examining skin lesions, like physical examinations, which require specialised expertise. This application will outline a cancer classification AI subsystem and skin disease identification subsystem designed to improve early recognition of skin disease.',
  },
  {
    id: 2,
    title: 'Technical Approach',
    body: 'This project employs a ResNet-50 model for binary skin cancer classification and a ResNet-152 model trained on the ISIC 2019 dataset for multi-class skin disease identification. Images pass through a standardised preprocessing pipeline before being fed to each model to ensure consistent and reliable inference.',
  },
];

/* Static data for Team Skana members. */
const TEAM_MEMBERS = [
  {
    id: 1,
    name: 'David Barboza',
    role: 'Main Application Developer',
    description:
      'Responsible for developing the main application and orchestration of AI agents and deep learning inference.',
  },
  {
    id: 2,
    name: 'Mohammad Taha',
    role: 'Machine Learning Developer',
    description:
      'Responsible for designing, developing and training several vision-based neural network architectures as well as fine-tuning and enhancing model performance.',
  },
  {
    id: 3,
    name: 'Karl Napod',
    role: 'Project Coordinator & Researcher',
    description:
      'Responsible for managing and coordinating the overall project with in-depth research, as well as creating core foundational documents to support the development and design of the project.',
  },
];



/* The following function is responsible for rendering a single project
   information card with a title and descriptive body text. */
function ProjectCard({ title, body }) {

  return (
    <div className="project-card">
      <span className="project-card-title">{title}</span>
      <p className="project-card-body">{body}</p>
    </div>
  );

}


/* The following function is responsible for rendering a single team member
   card displaying the person's name, role, and responsibility description. */
function TeamMemberCard({ name, role, description }) {

  return (
    <div className="team-member">
      <span className="team-member-name">{name}</span>
      <span className="team-member-role">{role}</span>
      <p className="team-member-desc">{description}</p>
    </div>
  );

}


/* The following function is responsible for rendering the About Us page in a
   book-style two-panel layout, with project information on the left and team
   member details on the right, separated by a vertical divider. */
function AboutPage() {

  return (
    <div className="about-page">

      <div className="about-book">

        <div className="about-panel">
          <p className="about-section-eyebrow">The Project</p>
          <h2 className="about-section-heading">DermAI by Skana</h2>
          <div className="project-cards-list">
            {PROJECT_CARDS.map((card) => (
              <ProjectCard key={card.id} title={card.title} body={card.body} />
            ))}
          </div>
        </div>

        <div className="about-divider" aria-hidden="true" />

        <div className="about-panel">
          <p className="about-section-eyebrow">The Team</p>
          <h2 className="about-section-heading">Team Skana</h2>
          <div className="team-members-list">
            {TEAM_MEMBERS.map((member) => (
              <TeamMemberCard
                key={member.id}
                name={member.name}
                role={member.role}
                description={member.description}
              />
            ))}
          </div>
        </div>

      </div>

    </div>
  );

}

export default AboutPage;
